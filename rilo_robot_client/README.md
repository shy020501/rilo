# Robot PC client

웹앱에서 자동으로 연결할 때는 다음 순서로 준비합니다.

1. 이 폴더를 로봇 PC의 `~/rilo_robot_client`에 복사하고, 사용할 Python 환경에 `requirements.txt`를 설치합니다.
2. 웹앱 원격 프로젝트에 SSH 접속 정보와 client 실행 환경을 저장합니다. 모의 검증은 `mock`, 실제 장비는 작성한 `module:factory` 브리지와 실제 환경 JSON 경로를 지정합니다.
3. **환경 연결**을 누르면 SSH가 client를 자동 실행하고 토큰을 stdin으로 전달합니다. 이 경로는 별도 `config.json`을 원격 PC에 만들지 않습니다.
4. 등록 후 별도의 **실행** 명령을 내려야 정책 동작이 시작됩니다. `probe` 모드는 연결 진단만 제공하므로 장비 실행 버튼이 활성화되지 않습니다.

실제 장비의 관측·실행기 API는 제공된 코드만으로 확정할 수 없어 브리지 구현이 필요합니다. 아래의 `examples.rb_env_adapter`는 그 구현을 위한 템플릿이며 그대로 선택해 장비를 움직이는 완성 드라이버가 아닙니다. 웹앱에서는 훅을 모두 구현한 실제 `module:factory`를 지정하세요.

이 폴더 전체를 로봇 PC의 `~/rilo_robot_client`로 복사해서 설치합니다. RILO 서버 소스, GPU, Torch는 필요하지 않습니다. Python 3.10 이상과 NumPy·HTTPX·Pillow만 사용합니다.

기존 로봇 환경은 `hilserl`이었던 것으로 알려져 있습니다. 해당 환경이 실제 설치된 환경인지 먼저 확인한 뒤 사용하세요. 로봇 SDK와 ROS 설정을 유지하려면 기존 환경을 쓰는 편이 좋습니다.

```bash
conda activate hilserl
cd ~/rilo_robot_client
python -m pip install -r requirements.txt
```

모의 클라이언트만 별도로 확인한다면 새 가상환경도 사용할 수 있습니다.

```bash
cd ~/rilo_robot_client
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.json config.json
chmod 600 config.json
```

`config.json`의 `project_token`에 해당 프로젝트의 토큰을 넣습니다. 토큰이 들어 있는 파일을 다른 사람에게 공유하지 마세요. 기본 서버 주소는 SSH 역방향 전달을 사용하는 `http://127.0.0.1:45587/robot/v1`입니다. 제어 주기 기본값은 10 Hz이며 실제 브리지의 등록 주기와 같아야 합니다. 추론 요청 제한 시간은 180초이고, 별도 heartbeat와 15초 실행 권한 확인이 유지됩니다. 연결 검사는 로봇을 움직이지 않습니다.

```bash
python -m rilo_robot_client --config config.json --probe
```

`--probe`가 기본 모드이며 한 번의 HTTP 상태 조회만 수행합니다. 토큰 인증이나 로봇 준비 완료를 뜻하지 않습니다. 실행 모드는 명시적으로 선택합니다. `--mock`은 이 폴더의 모의 환경을 등록하고 웹앱의 실행 명령을 기다립니다. 실제 하드웨어를 import하지 않습니다.

```bash
python -m rilo_robot_client --config config.json --mock
```

실제 환경은 명시적인 브리지 구현과 실제 환경 JSON 경로가 필요합니다.

```bash
python -m rilo_robot_client --config config.json \
  --adapter examples.rb_env_adapter:create_adapter \
  --env-config /실제/환경/설정.json
```

`examples/rb_env_adapter.py`는 제공된 `rb_env` builder와 `ChunkInterpreter` import를 사용하는 연결 템플릿입니다. 알 수 없는 `run_episode` 내부 메서드는 호출하지 않습니다. `adapter_options.bridge_factory`에 아래 훅을 구현한 `module:factory`를 지정해야 하며, 없으면 장비 생성 전에 중단합니다. 환경 등록 자체에서 reset·step·이동 명령을 호출하지 않습니다. 장비 SDK 생성자가 자체적으로 움직이는 구성은 브리지 작성자가 먼저 제거해야 합니다.

웹앱의 자동 실행은 `--agent-stdin` 모드를 사용합니다. SSH stdin 첫 줄의 `{ "client_config": {...}, "mode": "probe|mock|robot", "adapter": ..., "env_config": ... }`를 읽고, 토큰을 argv·환경변수·원격 설정 파일에 저장하지 않습니다. 등록이 끝나면 `RILO_CLIENT_READY` 한 줄을 보냅니다. 이 모드의 `probe`는 등록과 heartbeat만 유지하는 진단 클라이언트이며 로봇 실행 기능을 제공하지 않습니다. stdin이 닫히면 큐를 비우고 hold 후 종료합니다.

## 기존 실행기 연결

`ExecutionBridge`는 기존 비동기 실행기를 교체하지 않습니다. 기존 실행기의 동작에 맞춰 다음 훅을 제공하세요.

| 훅 | 역할 |
| --- | --- |
| `describe()` | 실제 측정 상태·카메라·행동 스키마 및 지원 기능 설명 |
| `observe()` | 측정값과 RGB 이미지로 `Observation` 생성 |
| `sample_tick()` | hold 상태에서 다음 실제 제어 틱의 새 관측을 반환해 필요한 초기 이력 확보 |
| `replace_pending(actions, epoch=...)` | 이전 대기 큐를 제거하고 해당 실행 세대의 명령만 채택 |
| `execute_tick(epoch=...)` | 제어 틱 하나를 실행하고 실제 적용 행동·다음 관측·보상을 반환 |
| `clear_pending(reason)` | 아직 적용하지 않은 명령 제거 |
| `hold(reason)` | 장비에 검증된 정지·유지 동작 수행 |
| `set_phase(phase)` | task/reset 단계에 맞는 관측 및 성공 조건 선택 |
| `safe_reset()` | 물체 복구와 로봇 후퇴·정지 상태를 독립적으로 확인 |
| `arm_reset()` | 안전 확인 후 장비의 명시적 arm reset을 한 번 수행하고 실제 후속 관측 반환 |
| `close()` | 기존 실행기와 장비 리소스 정리 |

`HookExecutionBridge`에 각 훅을 전달하면 기존 executor를 감쌀 수 있습니다. 기존 `ChunkInterpreter`는 `HookChunkDecoder`로 연결합니다. HTTP 계층은 모델의 전체 원시 청크를 넘기고, 해석기는 행동 표현·단위·좌표계를 확인한 뒤 `DecodedAction` 목록으로 변환합니다. 임의의 첫 번째 행 선택이나 장비에 맞지 않는 좌표 변환은 하지 않습니다.

DP task 정책은 현재 실행 청크의 끝에서 한 틱을 남겼을 때 실제 관측으로 다음 청크를 비동기 요청합니다. 요청은 최대 하나이며 실행 세대가 바뀌면 폐기합니다. 해석기에는 전체 원시 청크와 `request_control_tick`, `current_control_tick`, `observation_delay_ticks`를 전달합니다. 기본 벡터 해석기는 한 행이 한 제어 틱인 계약에 따라 이미 지난 행을 제외하고, 실제 사용한 원시 행 번호도 전이에 남깁니다. 사용자 `ChunkInterpreter`를 연결하면 이 시간 정렬도 해당 해석기가 담당합니다. 응답이 아직 없으면 명시적인 hold 상태로 기다립니다. GPT/reset에는 이 선행 요청을 적용하지 않습니다.

관측 키, 카메라 이름, 조인트 순서, hold 명령, 큐 교체 방법은 로봇마다 다릅니다. 클라이언트는 없는 측정값을 0으로 채우거나 0 벡터를 안전한 hold로 가정하지 않습니다. `replace_pending`만으로 큐 전체가 독립적으로 이동하면 안 됩니다. 각 실제 적용 틱은 `execute_tick`의 실행 권한 확인과 journal 기록을 통과해야 합니다. 기존 executor를 쓸 때에는 이 제어 지점을 연결하세요. 동기 SDK를 사용한다면 훅 내부에서 유한 시간 내에 중단 가능한 워커를 제공해야 합니다. Python await 취소만으로 실제 이동이 중단되지는 않습니다.

## 기록과 재시도

`ObservationHistory`는 HTTP 요청 횟수가 아니라 실제 제어 틱에 따라 관측을 저장합니다. 초기 hold 틱도 실제로 측정한 경우에만 이력에 포함합니다. 아직 확보하지 못한 과거 관측은 복제하지 않고 누락으로 보고합니다. 서버에 보낼 때는 현재 정책 계약이 요구하는 관측 수만 전송하며 프로토콜 한도는 8개입니다.

로컬 SQLite journal은 실행 직전의 식별자와 실행 후 실제 전이를 디스크에 기록합니다. 서버 응답을 잃어도 같은 전이 업로드를 재시도할 수 있고, 이미 시작된 이동을 다시 실행하지 않습니다. 실행 직후 프로세스가 죽어 완료 기록이 없는 경우에는 이동 결과를 알 수 없으므로 자동 재개하지 않습니다. 장비 상태를 확인하고 새로운 실행을 시작해야 합니다. 기록 파일을 지워 중복 실행 방지를 우회하지 마세요.

통신 실패, 만료된 실행 권한, Pause/Stop, 실행 세대 변경 시에는 대기 큐를 비우고 명시적 hold 훅을 호출해야 합니다. 서버 복귀나 늦게 도착한 추론 결과만으로 이전 명령을 재개하지 않습니다.

Residual 보정은 각 틱의 최신 측정값으로 요청합니다. 온라인 학습을 지원한다고 등록한 브리지는 다음 base action도 실제 청크에서 얻어야 합니다. 청크 경계에서는 전이를 먼저 디스크에 기록하고 다음 청크를 받은 뒤 최초 업로드 전에 보강합니다. 다음 base action을 얻지 못하면 hold하며, Stop 시에는 실제 전이를 `learning_pending`으로 보내 학습에서 제외합니다. 종료가 확정된 전이만 `bootstrap_unused`로 표시합니다. 마지막 행동을 다음 행동으로 꾸며 넣지 않습니다.
