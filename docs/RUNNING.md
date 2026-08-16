# 실행 방법과 내부 동작

카메라 앞의 사람을 K1/G1 로봇 목표 자세로 바꿔 실시간 표시하고, 다운스트림 정책이 쓸
참조 모션을 내보내는 시스템. 이 문서는 **어떻게 돌리는가**와 **안에서 무슨 일이
일어나는가**를 다룬다. 측정값과 실패 이력은 `PLAN.md`, 앞으로 할 일은 `todo.md`.

---

## 1. 빠른 실행

표시 방법이 둘이다. **로컬 GUI가 기본 권장**이고, 원격에서 봐야 할 때만 스트리밍을 쓴다.

### 로컬 GUI (권장)

```bash
cd ~/Projects/shape15

DISPLAY=:1 GEM-X/.venv/bin/python demo_webcam.py --flip --robot k1 --no_imgfeat
```

증류 네트워크가 **기본 경로**다 — `models/<robot>_retarget.pt`가 있으면 자동으로 쓰고,
없거나 `--ik`를 주면 PyRoki 솔버로 떨어진다. 다른 체크포인트는 `--mlp PATH`로 지정한다.

약 40초 뒤 창 두 개가 뜬다 — `GEM-X SOMA skeleton`(카메라+스켈레톤)과
`MuJoCo : ai_sapiens_23dof`(로봇). 카메라 창을 클릭해 포커스를 준 뒤:

| 키 | 동작 |
|---|---|
| `c` | 캘리브레이션 — 누르고 T-포즈를 ~1초 유지 |
| `q` | 종료 |

스트리밍보다 나은 점: MJPEG 인코딩·전송이 없어 워커 부하가 줄고(JPEG+put 1.5 ms),
MuJoCo 뷰어를 마우스로 돌리고 확대할 수 있으며, 브라우저 연결 문제가 없다.

### 브라우저 스트리밍

```bash
GEM-X/.venv/bin/python demo_webcam.py --flip --robot k1 --stream 8080 \
    --no_imgfeat --mlp models/k1_retarget_mlp_res.pt
```

**http://<이 머신 IP>:8080** 에 두 패널이 한 페이지로 뜨고, 하단 **Calibrate** 버튼이
`c` 키와 같은 일을 한다. 로봇 패널은 8081에서 따로 송출된다(페이지가 합쳐 보여준다).

정지: `pkill -9 -f demo_webcam.py; pkill -9 -f ik_server.py`

### 주요 옵션

| 플래그 | 뜻 |
|---|---|
| `--robot k1\|g1` | 대상 로봇. K1이 목표 로봇, G1은 참고용 |
| `--mlp PATH` | 다른 증류 체크포인트 지정. 기본은 `models/<robot>_retarget.pt` |
| `--smooth HZ` | 관절각 저역통과. 기본 2.0 — 정확도를 깎지 않고(9.01→8.93°) 최악 점프를 238→108°로 줄인다. `0`이면 끔 |
| `--ik` | 증류 네트워크 대신 PyRoki 솔버 강제 |
| `--kp2d_mlp PATH` | GEM denoiser·FK를 건너뛰는 학생(~30 Hz, 정확도는 손해) |
| `--no_imgfeat` | SAM-3D-Body 끄기. **화면 멈춤의 주원인이라 라이브에선 권장** |
| `--stream PORT` | 브라우저로 송출(카메라 PORT, 로봇 PORT+1). **생략하면 로컬 창** |
| `--flip` | 화면 좌우 반전(거울처럼 보이게) |
| `--mirror` | 로봇 좌우 반전. 거울 데모 전용, `--mlp`와 배타 |
| `--frames` | 방향 타겟 켜기. **측정상 더 나빠서 기본 꺼짐** |
| `--motion_command out.npz` | 50 Hz 정책 참조 저장. **Calibrate 시점부터** 종료까지 |
| `--save_raw DIR` | 촬영 원본 보존: `rgb.mp4`(오버레이 전 카메라 화면) + `soma.npz`(3D 스켈레톤·신뢰도·관절회전·하류로 보낸 타겟·프레임 타임스탬프). 촬영은 재현이 안 되므로 기본으로 켜고 찍을 것 |

### 오프라인 도구

```bash
# 지연·프레임간격 측정 (라이브와 동일 경로)
GEM-X/.venv/bin/python replay_delay.py <clip.mp4> out.mp4 --ik k1 \
    --ik_async --ik_render 8091 --profile

# 학습 데이터 만들기 → 학생 학습 → 오프셋 적합
GEM-X/.venv/bin/python make_labels.py a.mp4 b.mp4 --out data/x_k1.npz --robot k1
GEM-X/.venv/bin/python make_synth.py data/big4_g1.npz.perception.npz \
    --out synth --clips 8 --mode interp --blend 0.6
.venv-ik/bin/python distill.py data/x_k1.npz --features pose --width 1024 --arch res
.venv-ik/bin/python make_offsets.py data/x_k1.npz

# 배포 모델 재현 (부위별 전문가, 다리는 다리 입력만)
.venv-ik/bin/python distill.py data/big4_k1.npz --features pose --width 1024 \
    --arch parts --part_in solo --extra data/synth_interp_k1.npz --mirror \
    --model models/k1_retarget_solo.pt

# 사람 / 교사 / 모델들 나란히 렌더 — 수치만 보고 결론 내리지 말 것
GEM-X/.venv/bin/python render_labels.py data/legstatic_test_k1.npz outputs/x.mp4 \
    --models models/k1_retarget_parts.pt models/k1_retarget_tight.pt
```

`--arch parts` 옵션:

| 플래그 | 뜻 |
|---|---|
| `--part_in wide` | 다리 전문가가 torso 13관절 전부를 본다 (쇄골·머리 포함). 구 기본값 |
| `--part_in tight` | 척추(1-3)만 다리에, 쇄골(11,39)·머리(4-10)는 팔·몸통에만 |
| `--part_in solo` | 다리는 다리 관절만 본다. **골반 위 섭동에 다리 변화 0.000°** — 현재 기본 모델 |
| `--fuse` | 조립된 23각도 위에 보정 헤드. 측정상 효과 없음(−0.02°), 기록용으로 유지 |

### 회귀 검사

```bash
.venv-ik/bin/python test_apose.py          # 대칭 rest 자세 → 대칭 해가 나오는가
.venv-ik/bin/python test_calibrate.py k1   # Calibrate가 체형을 다시 재는가
.venv-ik/bin/python motion_command.py      # 필터·속도 자기검증
```

---

## 2. 프로세스 구조 — 왜 둘로 나뉘어 있나

```
┌─ demo_webcam.py ────────────┐        ┌─ ik_server.py ──────────────┐
│  GEM-X/.venv (torch, GPU)   │ stdin  │  .venv-ik (JAX/pyroki, CPU) │
│                             │ ─────► │                             │
│  카메라 → VitPose → GEM     │  728B  │  스케일·floor 캘리브레이션  │
│  → SOMA 자세 → 증류 MLP     │ ◄───── │  → 자세 확정 → MuJoCo 렌더  │
│  → 스켈레톤 오버레이 :8080  │ 96B    │  → 로봇 패널 :8081          │
└─────────────────────────────┘        └─────────────────────────────┘
```

두 가상환경이 **서로 배타적**이라 한 프로세스에 못 담는다 — 인지는 torch가, IK는
JAX/pyroki가 필요하고 버전이 충돌한다. 나눈 결과 부수 이득이 세 가지 있다: IK가 CPU에서
돌아 GPU를 인지와 다투지 않고, warp와 torch의 CUDA 그래프 충돌이 사라지며, 워커가 느려도
인지가 멈추지 않는다(아래 `IKLink`).

### 파이프 프로토콜

고정 크기 little-endian float32. 프레임당 한 번:

```
인지 → 워커 (728 B, --mlp면 +ndof*4)
  [0:42]    14개 관절 3D 위치 (카메라 프레임, 미터)
  [42:56]   14개 신뢰도 — 첫 값의 부호비트가 Calibrate 요청을 겸함
  [56:182]  14개 관절 회전 3x3 (--soma_rot 아니면 0)
  [182:]    (--mlp) 네트워크가 낸 관절각 ndof개

워커 → 인지 (4 + ndof*4 B)
  [0]       솔브 소요 ms
  [1:]      최종 관절각
```

캘리브레이션 요청을 **부호비트**로 보내는 이유: 고정 크기 프로토콜을 바꾸지 않으면서
값 자체도 보존되기 때문(워커가 `abs()`를 취한다).

---

## 3. 인지 파이프라인 (GPU, 루프를 막는 쪽)

프레임마다 순서대로:

1. **`Camera`** — 백그라운드 스레드가 최신 프레임만 유지. 루프는 항상 가장 새 프레임을
   읽는다(밀린 프레임을 따라가지 않는다). 카메라가 끊기면 재연결한다.
2. **VitPose** (24.5 ms) — SOMA 규격 77개 2D 키포인트. bbox는 이전 프레임 키포인트에서
   전파하고, 놓치면 전체 프레임에서 재검출한다.
3. **SAM-3D-Body 토큰** (선택) — 이미지 특징. 한 번에 ~280 ms라 **백그라운드 스레드**에서
   돌고 루프는 가장 최근에 끝난 토큰을 쓴다. 잘린 몸에서 정확도를 크게 올리지만
   (7.95% → 4.82%) 주기적 스터터의 주원인이라 라이브에선 `--no_imgfeat` 권장.
4. **GEM denoiser** (16.5 ms) — 64프레임 슬라이딩 윈도로 SOMA 자세 파라미터를 추정.
   그 중 마지막 프레임만 쓴다.
5. **SOMA FK** (6.2 ms) — 관절 3D 위치와 화면 투영. 오버레이와 워커 페이로드에 쓰인다.
6. **리타겟** — 아래 3가지 중 하나.

`--kp2d_mlp` 모드에서는 3·4·5가 **아예 없다** (GEM 모델을 로드조차 안 한다).

### 리타겟 세 경로

| 경로 | 무엇을 하나 | 교사 대비 오차(K1) | 루프 |
|---|---|---|---|
| PyRoki IK (기본) | 14개 점을 타겟으로 미분 IK를 워커에서 풀이 | 26.75° | ~57 ms |
| **증류 MLP** (`--mlp`) | SOMA articulation 228차원 → 관절각. CPU 0.3 ms | **10.71°** | ~57 ms |
| kp2d 학생 (`--kp2d_mlp`) | VitPose 2D 윈도 → 관절각. denoiser·FK 생략 | (G1 15.56°) | ~28 ms |

증류 MLP가 이기는 이유는 용량이 아니라 **정보**다. 14개 점은 팔의 비틀림 같은 자유도를
결정하지 못하는데, `body_pose`(SOMA 76관절 회전)는 그걸 담고 있고 라이브에서 공짜로
나온다. 자세한 실험은 `PLAN.md`.

---

## 4. 워커 (CPU)

1. **카메라→로봇 프레임 변환** — 축 치환이다(카메라 x우/y하/z전 → 로봇 x전/y좌/z상).
   부호 뒤집기가 아니라서 틀리면 로봇이 바닥 아래 눕는다.
2. **체형 스케일링** (`scale_to_robot`) — 사람 뼈 길이를 로봇 뼈 길이로 치환한다.
   세션 시작 15프레임에서 측정해 고정하고, Calibrate를 누르면 다시 잰다.
   재측정 중에는 **직전 floor를 유지**한다 — 0으로 두면 로봇이 0.7 m 떠올랐다가
   떨어지고 솔버가 나쁜 분기에 갇힌다.
3. **자세 확정**
   - IK 모드: pyroki 미분 IK. 관절 한계, 신뢰도 가중, twist 널스페이스를 함께 푼다
   - MLP 모드: 네트워크 관절각을 그대로 쓰고, **베이스만 Kabsch로 맞춰** 로봇이 제자리에
     서게 한다 (0.3 ms)
4. **관절 오프셋** — SOMA와 로봇의 관절 영점 차이를 상수로 흡수. IK 경로에만 적용한다
   (MLP는 교사를 직접 학습했으므로 불필요).
5. **MuJoCo 표시** — 물리 없이 `qpos`만 설정하고 `mj_forward`. 스트리밍이면 EGL 오프스크린.
6. **정책 참조** (`--motion_command`) — 50 Hz로 리샘플, one-euro 필터, URDF 속도한계
   클램프 후 차분. 출력은 관절위치 23 + 속도 23 + torso 방향 6 = shape14 정책의 관측과
   정확히 일치한다. 녹화는 Calibrate 신호(`conf[0]` 부호비트)에서 버퍼를 비우고 다시
   시작한다 — 그 이전 프레임은 이전 사람의 뼈 스케일로 측정된 것이라 참조가 못 된다.

---

## 5. 멈추지 않게 만드는 장치들

화면이 얼어붙는 것은 이 프로젝트에서 반복적으로 문제였다. 지금 들어 있는 방어:

- **`IKLink` (논블로킹 IPC)** — 인지는 워커의 답을 **기다리지 않는다**. 리더 스레드가
  답을 받아두고 루프는 최신 것을 쓴다. 워커가 밀리면 그 프레임을 버린다(약 10%).
  이것 하나로 프레임 간격 최대 **4017 ms → 308 ms**.
- **`--no_imgfeat`** — SAM 토큰 버스트(~280 ms) 제거.
- **`torch.set_num_threads(2)`** — CPU 네트워크가 코어당 스레드를 잡으면 Isaac 같은
  이웃 작업과 경합해 **0.09 ms 짜리 forward가 60 ms**가 된다. 실측 700배.
- **`Camera` 스레드** — 밀린 프레임을 소비하지 않고 최신만 본다.
- **재검출 1패스** — 추적 실패 시 3패스(1.7 s)가 아니라 1패스.

판정은 처리량이 아니라 **프레임 간격**으로 한다. `replay_delay.py`가 median/p99/max를
출력한다.

---

## 6. 학습 파이프라인

```
영상 ──► make_labels.py ──► perception.npz ──┬─► 교사(soma-retargeter) ──► 라벨
        (인지, GPU)                          │
                                             └─► 학생 입력 (body_pose 228)

make_synth.py ──► 합성 perception.npz ──► 교사 ──► 합성 라벨
        (영상 불필요)

distill.py ──► IK 기준선 + 학생 학습 + 홀드아웃 평가
make_offsets.py ──► IK용 상수 오프셋
```

**교사**는 soma-retargeter(사내 리타게터)다. 느리지만 오프라인이라 상관없다. 학생은
그 출력을 흉내내 실시간으로 낸다.

지켜야 할 규칙 셋:

1. **홀드아웃은 클립 단위** — 같은 클립의 인접 프레임은 거의 같아서 프레임 단위로 나누면
   무의미한 숫자가 나온다. 항상 마지막 2클립을 뺀다.
2. **인지는 재현 불가능하다** — SAM 토큰 워커가 백그라운드 스레드라 어느 프레임에 어떤
   토큰이 붙는지 실행마다 다르다. 같은 영상으로 다시 만들면 다른 데이터셋이 되고 이전
   수치와 비교할 수 없다. 그래서 `data/`의 npz를 보존한다.
3. **합성은 `--mode interp`로** — 관절 범위를 넓혀 균일 샘플링(`box`)하면 사람이 취하지
   않는 자세가 나오고, 교사가 그걸 관절 한계로 밀어 라벨의 41%가 망가진다. `interp`는
   실제 자세들의 블렌드라 매니폴드를 벗어나지 않는다(병리 0%).

---

## 7. 자주 겪는 문제

| 증상 | 원인과 조치 |
|---|---|
| `cannot open camera 0` | 이전 데모가 카메라를 붙잡고 있다. `pkill -9 -f demo_webcam.py` 후 3초 대기 |
| 모델 로드 시 `Error(s) in loading state_dict` | 체크포인트 구조와 코드가 불일치. `build_student()`가 `arch`/`blocks`를 읽으므로 최신 코드인지 확인 |
| 화면이 주기적으로 멈춤 | `--no_imgfeat` 사용. 그래도면 `nvidia-smi`로 이웃 작업 경합 확인 |
| CPU 네트워크가 수십 ms | `torch.set_num_threads(2)` 누락 |
| GUI 창이 안 뜸 | `DISPLAY=:1`을 붙였는지 확인(`:0`은 이 머신에 없다). `xdpyinfo`로 살아있는지 확인 |
| 영상이 VS Code에서 안 열림 | `mp4v`로 저장된 것. H.264만 재생되므로 ffmpeg으로 변환. `avc1`은 이 OpenCV 빌드에 없어 조용히 빈 파일을 만든다 |
| 로봇이 바닥 아래 누움 | 카메라→로봇 프레임 변환 오류(축 치환) |
| K1 로봇이 어둡게 렌더됨 | K1 MJCF에 조명 설정이 없다. 표시만의 문제 |

---

## 8. 파일 지도

| | |
|---|---|
| `demo_webcam.py` | 인지 루프, 카메라, 스켈레톤 오버레이, 학생 실행, 워커 기동 |
| `ik_server.py` | 워커: 캘리브레이션·IK/Kabsch·MuJoCo·정책 참조 |
| `ik_retarget.py` | IK 정의(ik_map, 스케일링, twist 널스페이스, 방향 타겟) |
| `mjpeg.py` | 브라우저 스트리밍과 Calibrate 버튼 |
| `motion_command.py` | 50 Hz 정책 참조 생성 |
| `trim_clip.py` | 촬영 앞뒤 정지 구간 잘라내기 (`--show`로 속도 프로파일) |
| `replay_soma.py` | 저장된 원본을 워커에 다시 흘려보내 리타게팅 재실험 (재촬영 불필요) |
| `ref_stream.py` | 참조 프레임을 UDP로 발행. `--replay`로 카메라 없이 수신부 시험 |
| `live_play.py` | 컨테이너에 올려 Isaac을 스트림으로 구동 (파일 대신 실시간) |
| `make_labels.py` / `make_synth.py` / `distill.py` / `make_offsets.py` | 학습 파이프라인 |
| `replay_delay.py` | 지연·프레임간격 측정 |
| `test_apose.py` / `test_calibrate.py` | 회귀 검사 |
| `robot_target.py` | 교사(soma-retargeter) 래퍼 |
| `demo_mediapipe.py` | 대안 인지 경로(CPU, 22 ms, 정확도 낮음) |
| `data/` | 데이터셋 (README 참조) |
| `models/` | 학습된 학생 체크포인트 |
| `fixtures/` | 관절 오프셋, rest 자세 |


---

## 9. 실시간 심 데모 (Isaac)

파일 대신 스트림으로 정책을 구동한다. 정책이 읽는 건 전부
`reference.<배열>[frame_ids]`라서, 씨앗 클립으로 env를 만든 뒤 슬롯 하나를 매 프레임
덮어쓰고 `frame_ids`를 거기 고정하면 된다 — shape14 코드는 건드리지 않는다.

```bash
# 1) 수신부 (컨테이너). 기동에 60~90초 걸리므로 "viewport on"을 기다릴 것
docker cp live_play.py cyclo_lab_shape14_eval:/tools/
docker cp mjpeg.py    cyclo_lab_shape14_eval:/tools/
docker exec cyclo_lab_shape14_eval bash -lc 'cd /workspace/cyclo_lab_private && \
  ./third_party/IsaacLab/_isaac_sim/python.sh /tools/live_play.py \
    --checkpoint <ckpt.pt> --seed_clip /motions/take5b/take5b.npz --stream 8100'

# 2) 발행부 (호스트). 컨테이너가 host 네트워크라 localhost로 통한다
.venv-ik/bin/python ref_stream.py --replay outputs/clips/take5b/take5b.npz
```

화면은 `http://<이 머신 IP>:8100`.

측정값 (RTX 5090, 다른 컨테이너와 GPU 공유):

| 항목 | 값 |
|---|---|
| 루프 속도 | 49 Hz (정책 50 Hz) |
| 스트림 반영률 | 434 중 347 프레임 (80%) |
| 종료/낙상 | 0 |

`--render_every 3`이 기본이다. 매 스텝 렌더링하면 27 Hz로 떨어지고, 증상은 **느린
영상이 아니라 참조 프레임이 건너뛰어지는 것**으로 나타난다.

아직 남은 것: 발행부가 카메라에 직접 물려 있지 않다. `ref_stream.Publisher`를
`ik_server`의 `motion_command` 경로에 연결하면 카메라→심이 끊김 없이 이어진다.
