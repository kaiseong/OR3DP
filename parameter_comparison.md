# Parameter Comparison: RISE vs Wrist Camera vs PCM

OR3DP 폴더 기준으로 정리한 파라미터 비교표입니다.

---

## 1. Train Configuration

### 1.1 기본 설정

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| horizon | 20 | 20 | 20 | 예측 액션 시퀀스 길이 |
| n_obs_steps | 1 | 1 | 1 | 관측 스텝 수 |
| n_action_steps | 8 | 8 | 8 | 실행할 액션 스텝 수 |
| n_latency_steps | 0 | 0 | 0 | 지연 보정 스텝 수 |
| dataset_obs_steps | 1 | 1 | 1 | 데이터셋 관측 스텝 (n_obs_steps와 동일) |
| keypoint_visible_rate | 1.0 | 1.0 | 1.0 | 키포인트 가시율 임계값 |
| obs_as_global_cond | True | True | True | 관측을 전역 조건으로 사용 |

### 1.2 Policy

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| _target_ | RISEPolicy | RISEPolicy | PCMPolicy | 정책 클래스 |
| num_action | 20 | 20 | 20 | 출력 액션 수 (=${horizon}) |
| input_dim | **6** | **6** | **7** | 입력 포인트 차원 (XYZRGB vs XYZRGB+conf) |
| obs_feature_dim | 512 | 512 | 512 | 관측 특징 차원 |
| hidden_dim | 512 | 512 | 512 | 히든 레이어 차원 |
| nheads | 8 | 8 | 8 | 어텐션 헤드 수 |
| num_encoder_layers | 4 | 4 | 4 | 인코더 레이어 수 |
| num_decoder_layers | 1 | 1 | 1 | 디코더 레이어 수 |
| dropout | 0.1 | 0.1 | 0.1 | 드롭아웃 비율 |
| enable_c_gate | - | - | False | confidence gate 사용 여부 (PCM 전용) |

### 1.3 EMA

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| update_after_step | 0 | 0 | 0 | EMA 시작 스텝 |
| inv_gamma | 1.0 | 1.0 | 1.0 | EMA 감마 역수 |
| power | 0.75 | 0.75 | 0.75 | EMA 파워 |
| min_value | 0.0 | 0.0 | 0.0 | EMA 최소값 |
| max_value | 0.9999 | 0.9999 | 0.9999 | EMA 최대값 |

### 1.4 Dataloader

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| batch_size | 120 | 120 | 120 | 배치 크기 |
| num_workers | 24 | 15 | 23 | 워커 수 |
| shuffle | True | True | True | 셔플 여부 |
| drop_last | True | True | True | 마지막 배치 드롭 |
| pin_memory | - | True | - | 메모리 고정 |
| persistent_workers | - | True | - | 워커 유지 |

### 1.5 Val Dataloader

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| batch_size | 190 | 128 | 128 | 검증 배치 크기 |
| num_workers | 15 | 10 | 10 | 검증 워커 수 |
| shuffle | False | False | False | 검증 셔플 |
| drop_last | False | False | - | 마지막 배치 드롭 |

### 1.6 Optimizer

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| _target_ | AdamW | AdamW | AdamW | 옵티마이저 |
| lr | 3.0e-4 | 3.0e-4 | 3.0e-4 | 학습률 |
| betas | [0.95, 0.999] | [0.95, 0.999] | [0.95, 0.999] | Adam 베타 |
| eps | 1.0e-8 | 1.0e-8 | 1.0e-8 | Adam 엡실론 |
| weight_decay | 1.0e-6 | 1.0e-6 | 1.0e-6 | 가중치 감쇠 |

### 1.7 Training

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| device | cuda:0 | cuda:0 | cuda:0 | 학습 디바이스 |
| seed | 233 | 233 | 233 | 랜덤 시드 |
| debug | False | False | False | 디버그 모드 |
| resume | True | False | True | 체크포인트 재개 |
| lr_warmup_steps | 1000 | 1000 | 1000 | 학습률 웜업 스텝 |
| num_epochs | **60** | **1** | **120** | 에폭 수 |
| use_ema | False | False | False | EMA 사용 여부 |
| rollout_every | 1 | 1 | 1 | 롤아웃 주기 |
| save_epochs | 20 | 20 | 15 | 체크포인트 저장 주기 |
| tqdm_interval_sec | 1.0 | 1.0 | 1.0 | 진행바 업데이트 간격 |
| validation_every | 5 | 5 | 5 | 검증 주기 |
| translation (X) | [-0.142, 0.585] | [-0.142, 0.585] | [-0.142, 0.585] | X축 정규화 범위 (m) |
| translation (Y) | [-0.500, 0.330] | [-0.500, 0.330] | [-0.500, 0.330] | Y축 정규화 범위 (m) |
| translation (Z) | [-0.085, 0.470] | [-0.085, 0.470] | [-0.085, 0.470] | Z축 정규화 범위 (m) |

---

## 2. Shape Meta

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| pointcloud_shape | [92160, **6**] | [92160, **6**] | [92160, **7**] | 포인트클라우드 형상 [max_points, features] |
| action_shape | [7] | [7] | [7] | 액션 차원 (6DOF + gripper) |
| robot_obs_shape | - | - | [7] | 로봇 관측 차원 (6DOF + gripper, PCM 전용) |

---

## 3. Dataset Configuration

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| _target_ | RISE_RealStackPointCloudDataset | BlendingDataset | PCM_RealStackPointCloudDataset | 데이터셋 클래스 |
| use_cache | True | True | True | 캐시 사용 |
| voxel_size | 0.005 | 0.005 | 0.005 | 복셀 크기 (m) |
| downsample_factor | **3** | **6** | **4** | 다운샘플 비율 |
| downsample_use_all_offsets | False | False | False | 모든 오프셋 사용 |
| group_by_offsets | False | False | False | 오프셋별 그룹화 |
| seed | 233 | 233 | 233 | 랜덤 시드 |
| val_ratio | 0.0 | 0.0 | 0.0 | 검증 비율 |
| max_train_episodes | null | null | null | 최대 학습 에피소드 |
| enable_blending | - | **True** | - | Blending 활성화 (Wrist Camera 전용) |

### 3.1 Augmentation (RISE, Wrist Camera만 해당)

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| aug | False | False | - | 데이터 증강 활성화 |
| aug_trans_min | [-0.2, -0.2, -0.2] | [-0.2, -0.2, -0.2] | - | 이동 증강 최소 (m) |
| aug_trans_max | [0.2, 0.2, 0.2] | [0.2, 0.2, 0.2] | - | 이동 증강 최대 (m) |
| aug_rot_min | [-30, -30, -30] | [-30, -30, -30] | - | 회전 증강 최소 (deg) |
| aug_rot_max | [30, 30, 30] | [30, 30, 30] | - | 회전 증강 최대 (deg) |
| aug_jitter | False | False | - | 지터 증강 |
| aug_jitter_params | [0.4, 0.4, 0.2, 0.1] | [0.4, 0.4, 0.2, 0.1] | - | 지터 파라미터 |
| aug_jitter_prob | 0.2 | 0.2 | - | 지터 확률 |

---

## 4. Preprocessing Configuration

### 4.1 pc_preprocessor_config (공통)

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| enable_transform | True | True | True | 좌표 변환 활성화 |
| enable_cropping | True | True | True | Workspace crop 활성화 |
| enable_sampling | False | False | False | 샘플링 활성화 |
| target_num_points | 1024 | 1024 | 1024 | 타겟 포인트 수 |
| use_cuda | True | True | True | CUDA 사용 |
| verbose | False | False | False | 상세 로그 |

### 4.2 Filter 설정

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| enable_filter | **False** | **True** | **True** | 필터 활성화 |
| filter_type | - | **ror** | - | 필터 타입 (ror/sor) |
| nb_points | 10 | 10 | 10 | SOR 이웃 포인트 수 |
| sor_std | 1.7 | 1.7 | 1.7 | SOR 표준편차 배수 |
| ror_nb_points | - | 12 | - | ROR 이웃 포인트 수 |
| ror_radius | - | 0.01 | - | ROR 반경 (m) |

### 4.3 Temporal (PCM) 설정

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| enable_temporal | **False** | **False** | **True** | PCM (시간적 메모리) 활성화 |
| export_mode | - | - | fused | 출력 모드 (fused/memory/frame) |
| temporal_voxel_size | - | - | 0.005 | PCM 복셀 크기 (m) |
| temporal_decay | - | - | 0.95 | 시간 감쇠 계수 (confidence 계산) |

### 4.4 Occlusion Prune 설정 (PCM 전용)

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| enable_occlusion_prune | - | False | **True** | Occlusion prune 활성화 |
| depth_width | - | - | 320 | Depth 이미지 너비 (px) |
| depth_height | - | - | 288 | Depth 이미지 높이 (px) |
| K_depth | - | - | [[252.69, 0, 166.12], ...] | Depth 카메라 내부 파라미터 |
| erode_k | - | - | 1 | Erosion 커널 크기 |
| miss_prune_frames | - | - | 20 | 연속 미관측 제거 프레임 수 |
| miss_min_age | - | - | 2 | Miss prune 최소 나이 |

### 4.5 Extrinsics & Workspace (공통)

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| extrinsics_matrix | 4x4 행렬 | 4x4 행렬 | 4x4 행렬 | 카메라→베이스 변환 행렬 |
| workspace_bounds (X) | [-0.132, 0.715] | [-0.132, 0.715] | [-0.132, 0.715] | X축 작업 공간 (m) |
| workspace_bounds (Y) | [-0.400, 0.350] | [-0.400, 0.350] | [-0.400, 0.350] | Y축 작업 공간 (m) |
| workspace_bounds (Z) | [-0.100, 0.600] | [-0.100, 0.600] | [-0.100, 0.600] | Z축 작업 공간 (m) |

### 4.6 low_dim_preprocessor_config

| 파라미터 | RISE | Wrist Camera | PCM | 설명 |
|---------|------|--------------|-----|------|
| robot_to_base | [[1,0,0,-0.04], [0,1,0,-0.29], [0,0,1,-0.03], [0,0,0,1]] | 동일 | 동일 | 로봇→베이스 변환 행렬 |

---

## 5. Camera Configuration

### 5.1 Femto Bolt (SingleOrbbec)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| rgb_resolution | (1280, 720) | RGB 해상도 |
| put_fps | 30 | 캡처 FPS |
| get_max_k | 30 | 링 버퍼 크기 |
| mode | C2D | 정렬 모드 (Color to Depth) |
| depth_image size | (320, 288) | Depth 이미지 크기 |

### 5.2 D405 (SingleRealSense) - 현재 비활성화

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| resolution | (424, 240) | 해상도 |
| put_fps | 60 | 캡처 FPS |
| get_max_k | 60 | 링 버퍼 크기 |
| enable_threshold_filter | True | 거리 임계값 필터 |
| threshold_min_dist | 0.04 | 최소 거리 (m) |
| threshold_max_dist | 0.20 | 최대 거리 (m) |
| enable_spatial_filter | True | 공간 필터 |
| spatial_magnitude | 2 | 반복 횟수 (1~5) |
| spatial_smooth_alpha | 0.7 | 필터 강도 (0.25~1.0) |
| spatial_smooth_delta | 15 | Edge 감지 임계값 (1~50) |

---

## 6. Evaluation Configuration

### 6.1 eval_piper_RISE.py

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| frequency | 10 | 제어 주파수 (Hz) |
| action_offset | 0 | 액션 오프셋 |
| action_exec_latency | 0.02 | 실행 지연 (s) |
| inference_interval | 10 (cnt%10) | 추론 간격 (10회 루프당 1회) |
| timeout | 60000 | 평가 타임아웃 (ms) |
| orbbec_mode | C2D | Orbbec 정렬 모드 |
| grip_threshold | 55 | 그리퍼 임계값 (>55 → 85, else 0) |

### 6.2 eval_piper_PCM.py

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| frequency | 10 | 제어 주파수 (Hz) |
| action_offset | 0 | 액션 오프셋 |
| action_exec_latency | 0.01 | 실행 지연 (s) - RISE보다 작음 |
| inference_interval | 10 (cnt%10) | 추론 간격 (10회 루프당 1회) |
| timeout | 60000 | 평가 타임아웃 (ms) |
| orbbec_mode | C2D | Orbbec 정렬 모드 |
| robot_obs | 사용 | 로봇 관측값 사용 (10D: xyz+rot6d+gripper) |
| reset_temporal | 사용 | 에피소드 시작/종료 시 PCM 리셋 |

### 6.3 RISE vs PCM 평가 비교

| 항목 | RISE | PCM | 차이점 |
|-----|------|-----|--------|
| Policy 입력 | cloud_data만 | cloud_data + robot_obs | PCM은 로봇 상태도 입력 |
| action_exec_latency | 0.02s | 0.01s | PCM이 더 빠른 응답 |
| Temporal 리셋 | - | reset_temporal() 호출 | PCM은 에피소드마다 메모리 초기화 |
| Gripper 처리 | np.where(>55, 85, 0) | 연속값 그대로 사용 | RISE는 이산화, PCM은 연속 |
| LowDimPreprocessor | 미사용 | 사용 | PCM은 로봇 좌표 변환 필요 |

---

## 7. 주요 차이점 요약

| 항목 | RISE | Wrist Camera | PCM |
|-----|------|--------------|-----|
| **포인트 차원** | 6 (XYZRGB) | 6 (XYZRGB) | 7 (XYZRGB + confidence) |
| **필터** | 없음 | ROR | SOR (기본) |
| **시간적 메모리** | 없음 | 없음 | 있음 (PCM) |
| **Occlusion Prune** | 없음 | 없음 | 있음 |
| **로봇 관측** | 미사용 | 미사용 | 사용 (robot_obs) |
| **다운샘플** | 3 | 6 | 4 |
| **에폭** | 60 | 1 | 120 |
| **Blending** | 없음 | Femto+D405 | 없음 |

---

## 8. 파일 경로

| 설정 유형 | 파일 경로 |
|---------|----------|
| RISE Train | `/home/moai/OR3DP/pcdp/config/train_diffusion_RISE_cube_stack.yaml` |
| RISE Task | `/home/moai/OR3DP/pcdp/config/task/RISE_stack.yaml` |
| Wrist Camera Train | `/home/moai/OR3DP/pcdp/config/train_diffusion_Blending_cube_stack.yaml` |
| Wrist Camera Task | `/home/moai/OR3DP/pcdp/config/task/Blending.yaml` |
| PCM Train | `/home/moai/OR3DP/pcdp/config/train_diffusion_PCM_cube_stack.yaml` |
| PCM Task | `/home/moai/OR3DP/pcdp/config/task/PCM_stack.yaml` |
| RISE Eval | `/home/moai/OR3DP/eval_piper_RISE.py` |
| PCM Eval | `/home/moai/OR3DP/eval_piper_PCM.py` |
| Camera (Orbbec) | `/home/moai/OR3DP/pcdp/real_world/single_orbbec.py` |
| Camera (RealSense) | `/home/moai/OR3DP/pcdp/real_world/single_realsense.py` |
| Real Env | `/home/moai/OR3DP/pcdp/real_world/real_env_piper.py` |
