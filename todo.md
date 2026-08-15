# 재시작할 때 먼저 읽기 (2026-08-14 종료 시점)

컴퓨터를 껐다 켠 뒤 이어서 하는 순서. **자산은 전부 `data/`에 보존돼 있다.**

## 상태 한 줄

soma-retargeter를 PR#3으로 올려 K1 교사의 왼다리 결함을 고쳤고, K1 실데이터 재라벨링은
끝났다(병리 26% → **6.6%**). 합성 라벨링만 11/24에서 멈춰 있다.

## 바로 할 것

1. **합성 라벨링 이어서** — 부분결과를 tmp로 되돌린 뒤 재실행하면 11클립은 건너뛴다
   ```
   TMP=<이 세션의 작업 tmp>            # 새 잡이면 아무 쓰기 가능한 경로면 된다
   cp data/big4_g1.npz.perception.npz data/synth.perception.npz $TMP/
   cp data/partial_teacher_k1/*.npy $TMP/
   GEM-X/.venv/bin/python make_labels.py x --out $TMP/synth_k1.npz --robot k1 \
       --mid $TMP/synth.perception.npz --teacher      # 남은 13클립 ~15분
   cp $TMP/synth_k1.npz data/
   ```

2. **K1 기준선 재측정** (아래 2번 항목) — 합성을 기다릴 필요 없이 지금 바로 가능
   ```
   .venv-ik/bin/python distill.py data/big4_k1.npz --features pose --width 1024 \
       --model models/k1_retarget_mlp.pt
   .venv-ik/bin/python make_offsets.py data/big4_k1.npz
   ```

## 보존된 자산

| | |
|---|---|
| `data/big4_g1.npz.perception.npz` | 실영상 인지 9,396프레임 — **가장 비싸고 재현 불가** |
| `data/big4_k1.npz` | **새 K1 교사 라벨** (병리 6.6%) |
| `data/synth.perception.npz` | 합성 14,400프레임 |
| `data/partial_teacher_k1/` | 부분결과 26개 (실 15 + 합성 11) |
| `data/big4_g1.npz`, `synth_g1.npz` | G1 라벨 (참고용) |
| `models/g1_*.pt` | G1 모델 (참고용) |
| `fixtures/g1_joint_offsets.npz` | G1 오프셋. **K1 오프셋은 없다** — 재생성 필요 |

## 주의

- `fixtures/k1_joint_offsets.npz`가 없어 `ik_server.py --robot k1`은 오프셋 0으로 돈다.
  잘못된 오프셋보다 낫지만, 재측정 후 `make_offsets.py`로 만들 것
- `models/k1_retarget_mlp.pt`도 없다(오염 라벨 기반이라 삭제). 2번 실행 시 생성됨
- 인지는 재현 불가능하다(SAM 워커가 백그라운드 스레드) — `data/`의 perception npz를
  절대 다시 만들려 하지 말 것. 자세한 내용은 `data/README.md`

---

# TODO — 카메라에서 실기까지

**목표 로봇은 K1이다.** G1은 **현재 방법이 돌아가는지 보는 참고용일 뿐**이고 그 외의
이유는 없다. G1 모델을 다듬거나 G1 수치를 결과로 보고하지 않는다. 아이디어를 싸게
검증할 때만 쓴다(G1 교사가 ~100 fps, K1은 ~10 fps).

정확도 수치는 전부 동일 홀드아웃(마지막 2클립 `taeguk_1st` + `video.mov`, 4,697프레임)
기준, 학습은 나머지 13클립 4,699프레임. 배경은 `PLAN.md`.

---

## 1. K1 재라벨링 — 거의 완료

soma-retargeter를 PR#1 → PR#3으로 올린 뒤 다시 생성 중. 구 라벨은 왼다리가 50.4%
프레임에서 망가져 있었고(`outputs/k1_teacher_fix.mp4`) 전량 삭제했다.

- [x] 서브모듈 업그레이드 (soma-retargeter 8dc838e + ai_sapiens 6a2af37)
- [x] 오염 라벨·모델·오프셋 삭제
- [x] **실데이터 15/15 클립** 완료
- [x] 실데이터 15/15 → `data/big4_k1.npz` (병리 26% → 6.6%)
- [ ] **합성 라벨링** 진행 중 (재개됨)
- [ ] 끝나면 `data/`로 복사 — tmp는 잡 삭제와 함께 사라진다
  ```
  cp $TMP/big4_k1.npz $TMP/synth_k1.npz data/
  ```

## 2. K1 기준선 재측정

옛 수치(IK 28.70°, MLP 15.33°)는 오염 라벨 기준이라 전부 무효. 처음부터 다시.

- [x] **IK 기준선 + MLP** — IK 26.75° / MLP plain 14.94° / **MLP res 13.97°**
  ```
  .venv-ik/bin/python distill.py data/big4_k1.npz --features pose --width 1024 \
      --model models/k1_retarget_mlp.pt
  ```
- [x] **오프셋 재생성** — 단 7/23 관절에만 적용됨(구 22/23). 교사가 고쳐지자 상수로
  설명되는 성분이 사라졌다는 뜻
- [x] ~~오프셋 재생성~~ — 지금 `fixtures/k1_joint_offsets.npz`가 없어 워커는 오프셋 0
  ```
  .venv-ik/bin/python make_offsets.py data/big4_k1.npz
  ```
- [x] **병리율 확인** — 전체 6.6%, 테스트셋 0.5%, clip16 0.0% / clip17 1.0%
- [x] **렌더로 눈 검증** → `outputs/k1_relabeled_check.mp4` (사람 vs 새 교사, 3구간 27초)
- [x] `test_apose.py` 재실행 — **비대칭 22.3°가 그대로다.** 예상과 달리 이 결함은
  교사가 아니라 **우리 IK 자체**의 것이었다 (4번 아래 항목으로 이동)

## 3. K1 합성 데이터 → 라벨링 → 재측정

- [x] **실 vs 실+합성** — box 방식은 무효(병리 41.3%), `--mode interp` 신설로 해결
- [x] **blend 스윕** — 0.6이 최적: **14.26 → 10.79°**
- [x] **양 확대는 역효과** — 8클립(1:1) 11.74° vs 24클립(3:1) 12.18°
- [x] **부위별 전문가 구조** — 정확도 동등, 단 파라미터 32% 절감(엣지용 카드)

## 4. K1 정확도 올리기

- [x] **residual + LayerNorm** — `distill.py --arch res`로 편입 완료.
  새 라벨에서 재확인: plain 14.94° → **res 13.97°**
- [ ] **왼무릎 비대칭** — 우리 IK의 결함. `test_apose.py`에서 좌 22.3° vs 우 0.0°
- [x] **K1 MLP 최종** → `models/k1_retarget_mlp_res.pt` **10.71°** (IK 26.75° 대비 −16.0°)
- [ ] **K1 kp2d 학생** — denoiser+FK를 잘라 ~30 Hz. G1에서 MLP 11.48° vs kp2d 15.56°
  (−4.1°). K1은 출발점이 더 어려워 손실이 클 것이므로, **속도가 실제로 필요할 때만**

## 5. 라이브 — target (카메라 → MuJoCo에 목표 자세 표시)

**지금 동작하는 단계.** 남은 건 K1 모델 배선과 화면 품질.

- [ ] K1 MLP 라이브 확인 (체크포인트가 로봇/DOF를 담고 있어 코드 변경 불필요)
  ```
  GEM-X/.venv/bin/python demo_webcam.py --flip --robot k1 --stream 8080 \
      --no_imgfeat --mlp models/k1_retarget_mlp.pt
  ```
- [ ] **화면 멈춤 확인** — 사용자가 반복 지적한 항목. 처리량이 아니라 **프레임 간격**으로
  판정하고(`replay_delay.py`가 median/p99/max 출력) 반드시 눈으로 볼 것.
  원인 순서: SAM-3D-Body 토큰 버스트(~280 ms, `--no_imgfeat`로 제거) → Isaac 경합 →
  추적 실패 재검출
- [ ] 다중 인물 — 추적이 bbox 전파뿐. YOLOX+ByteTrack(62 ms/frame) 가능해진 상태

## 6. 라이브 — sim (카메라 입력으로 shape14 정책을 sim에서 구동)

`shape14/HANDOFF_ARBITRARY_TARGET.md` 기준. **인터페이스가 이미 맞는다**:

> student 관측 124차원 = `motion_command(46)` + `motion_anchor_ori_b(6)` +
> `base_ang_vel(3)` + `joint_pos_rel(23)` + `joint_vel_rel(23)` + `last_action(23)`

앞의 46+6이 정확히 우리 `motion_command.py` 출력이다(관절위치 23 + 속도 23 + torso 6,
50 Hz). 나머지는 로봇 자신의 IMU/엔코더.

- [ ] **shape14의 teacher→student 증류 상태 확인** — teacher는 미래 2초를 보므로 배포
  불가. `Cyclo-Mimic-K1-Rev1-Multi-Student` 태스크는 준비돼 있으나 **성능 이전 여부가
  미확인**이고, shape14 문서가 이것을 그쪽 1순위로 지목하고 있다
- [ ] **feasibility 게이트** — shape14는 관절속도 114 rad/s 클립을 학습에서 제외했고,
  안전 클립은 4.6~12.6 rad/s였다. **우리 출력은 max 20.9 rad/s**(URDF 한계로 클램프한
  값)라 경계에 있다. sim에 넣기 전에 우리 쪽에서 더 조일지 판단
- [ ] `motion_command.py`의 npz 출력을 shape14가 읽는 CSV 형식으로 내보내는 어댑터
  (우선 파일 경유로 오프라인 검증 → 그 다음 스트리밍)

## 7. 라이브 — real (카메라 입력으로 실기 배포)

`shape14` + `Projects/ai_sapiens_private`.

- [ ] **스트리밍 입력 배선** — C++ 런타임의 `MotionReference`가 생성자에서 CSV 전체를
  읽는다(`motion_reference.cpp:load_motion_csv`). 그 자리에 ROS 토픽 구독자를 넣어야
  한다. 인터페이스는 좁다: `seek(float)`, `joint_pos()`, `joint_vel()`,
  `root_quaternion()`, `joint_order()`
- [ ] **프레임 유실 대책** — `joint_vel`이 전진차분이라 프레임이 빠지면 속도가 튄다.
  버퍼링 + 보간 + 타임아웃 시 홀드. (우리 `motion_command.py`가 이미 보간·클램프를
  하므로 어디서 책임질지 정할 것)
- [ ] 안전: feasibility 게이트를 실기 전에 반드시

## 8. 라이브 — real, 엣지 (로봇 카메라 입력으로 온보드 실행)

- [ ] 현재 인지 비용: VitPose 24.5 ms + GEM predict 16.5 ms + FK 6.2 ms (RTX 5090).
  **엣지에서 이 예산이 성립하는지가 관문** — 성립 안 하면 kp2d 학생(4번)이 필수가 되고,
  VitPose 경량화도 검토 대상(SOMA-77 호환이 깨지는 문제 있음)
- [ ] 로봇 카메라 파라미터·프레이밍에 맞춘 캘리브레이션(웹 버튼 경로 재사용)
- [ ] ONNX/TensorRT — 측정상 PyTorch fp16보다 **느렸으므로** 엣지 하드웨어에서 다시 잴 것

---

## 하지 말 것 (측정으로 기각됨)

- **학습 더 돌리기** — epoch 12에서 test 최저, 600까지 가면 train 0.12°로 암기만 는다
- **폭/깊이 무작정 키우기** — plain 구조는 w4096·6층에서 오히려 악화 (residual은 예외)
- **시간 컨텍스트(과거 프레임 쌓기)** — 15.4 → 15.2°, 노이즈 수준
- **soma-retargeter 후처리 강제 ON** — 병리 1.0% → 44.4%로 악화. 원작자 설정이 맞다
- **FK 스킵(보조 t14 출력)** — t14 오차 179.7 mm로, "평균 포즈 고정"(223.7 mm)보다 겨우
  나은 수준. identity·scale·global_orient를 다 줘도 안 움직인다
- **지연 예측기** — 132 ms 늦음의 비용이 +0.28°인데 예측기는 1.6~2.5°를 지불한다

## 위생

- [ ] `replay_delay.py` / `record_input.py`의 mp4v → VS Code(Chromium)는 H.264만 재생.
  avc1은 이 OpenCV 빌드에 없으므로 mp4v로 쓰고 ffmpeg 변환
- [ ] G1 라벨은 재생성하지 않는다(참고용이므로). 단 G1 수치를 인용할 때는
  `newton_pipeline.py`가 1,141줄 바뀐 구버전 라벨 기준임을 명시할 것
