# RILO

**Robotic Interaction and Learning Orchestrator** — GPT가 관측을 보고 수치 action을 직접 생성하는 로봇 실험 프로토타입.

```text
Web UI ↔ Runner ↔ Policy (OpenAI Responses API)
             ↕
       EnvironmentAdapter → LIBERO / 다른 simulator / 실제 로봇
             ↓
       관측·action·실행 결과 기록
```

## 실행

Linux / Python **3.10**, 이름이 `rilo`인 Conda 환경을 사용한다. 직접 수집은 시연 데이터 다운로드 없이 시작할 수 있고, 공식 LIBERO 시연을 가져와 함께 학습할 수도 있다.

```bash
git clone https://github.com/shy020501/rilo.git
cd rilo
conda create -n rilo python=3.10 pip
conda activate rilo
python -m pip install --upgrade pip
# NVIDIA CUDA 12.4 빌드 예시. 이미 적절한 PyTorch가 있다면 생략 가능.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[libero,dev]'
python scripts/setup_libero.py

cp .env.example .env
# .env에 OPENAI_API_KEY 설정 또는 셸에서 export
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-6-astra"
export LIBERO_SUITE="libero_90"
export RILO_TASK_ID=7
rilo
```

**http://127.0.0.1:8000** → 프로젝트 선택 또는 생성 → 수집 화면에서 **환경 연결** → 설정 확인 → **수집 시작** 또는 **한 번만 실행**.
환경 연결은 카메라 관측을 준비하며 run을 만들거나 GPT를 호출하지 않는다. 키 없이도 연결과 카메라 확인은 가능하다. 키는 서버에서만 읽으며, 변경하면 서버를 재시작한다.

- 메인 화면은 프로젝트별 demo 수, reset 결과, API 비용을 보여준다. 프로젝트 안에서 파이프라인 개요·수집 제어·episode 데이터·모델·설정을 확인한다.
- 프로젝트 생성·화면 이동·환경 연결은 수집이나 API 호출을 시작하지 않는다. **수집 시작 / 한 번만 실행**을 눌러야 run을 만들고 수집한다. 실행 중에는 **Pause**로 멈출 수 있다.
- 기존 API 실험은 환경·suite·task별 프로젝트에 자동 연결한다. 원본 run, 이미지, 비용 원장과 영상은 이동하거나 변경하지 않는다. 프로젝트 이름·목표·run 연결만 별도 `artifacts/projects.json`에 저장한다.
- **개요 → 전체 파이프라인 실행**을 누르면 실행 설정 팝업이 열린다. 누적 성공 episode 목표, BC 학습 step 수, 평가 횟수·성공률 목표와 RL 실행량을 확인하고 **설정 저장 후 실행**을 눌러 남은 단계를 시작한다. 현재 보유한 성공 데이터도 목표에 포함되며, 팝업을 열거나 취소하는 것만으로는 설정을 저장하거나 실행하지 않는다. **수집** 탭의 수집 실행 설정에서 누적 성공 episode 목표, run당 rollout 수, API 비용과 task·reset 실행 한도를 함께 관리한다. 초기 seed는 입력에서 숨기고 저장된 값을 유지한다. **데이터** 탭은 episode 조회·보관·삭제를, **모델** 탭은 BC·RL 학습 선택, 각각의 지표·기록과 모델 레지스트리를 표시한다. BC는 DP를 지원하며 pi0·pi0.5는 미구현으로 표시한다. RL은 Residual TD3를 지원하고 필요한 Progress 모델은 자동 준비한다. **평가** 탭은 결과·기록을 상단에 표시하고 개별 평가와 실시간 카메라를 제공한다. 학습·평가 실행 버튼을 누르면 해당 설정 팝업이 열리며, 확인 후 설정을 저장하고 실행한다. 평가의 Reset 방식은 수집 설정 따르기·GPT reset·환경 reset·안전 후퇴 후 초기화 중 선택하며, BC와 RL 평가 및 전체 파이프라인 평가에 공통 적용한다. **설정** 탭은 프로젝트 이름·설명, GPT 모델과 추론 수준, Task grasp 기준만 관리한다. 수집·일반 설정 저장은 실행을 시작하지 않으며, 학습·평가 팝업을 열거나 취소하는 것만으로도 실행되지 않는다. 기본 학습 장치는 `cuda:0`이며 CPU 실행은 모델 탭에서 `cpu`로 명시적으로 바꾼다.
- **모델 → BC 학습 지표·로그**의 학습 step·Train loss·Validation loss를 누르면 실제 기록을 자동 갱신하는 그래프 팝업이 열린다. BC 실행 팝업에서 **체크포인트 저장 간격**을 steps 단위로 설정할 수 있으며, 중간 모델·EMA·optimizer 상태를 저장한다. 검증 최상 `policy.pt`는 따로 유지한다. BC 디스크 체크포인트를 통한 학습 재개 기능은 아직 제공하지 않는다.
- 기존 상세 콘솔은 **/console**이다. 프로젝트 **데이터**에서 episode 행을 누르거나 키보드 Enter/Space로 열고 **영상 생성**을 누르면 해당 episode의 영상을 백그라운드로 만든다. 생성 상태·오류를 확인할 수 있고, 완료 후 같은 상세 화면에서 재생한다. **영상 삭제**와 **MP4 다운로드** 버튼은 같은 줄에 표시된다. 영상 삭제는 생성된 영상과 포스터만 지우며 episode의 관측·action·성공 기록은 유지한다. 저장된 이미지로만 변환하며 로봇 실행이나 GPT 호출은 발생하지 않는다. 실행 중인 run이나 해당 프로젝트의 파이프라인이 사용하는 데이터는 작업 종료 후 변환한다.
- 프로젝트 **설정**의 모델·추론 수준·episode 수·reset 방식·실행 한도는 **다음 새 run 시작 시점**에 함께 확정된다. 설정을 바꾸기 위해 환경을 다시 연결하거나 서버를 재시작할 필요는 없다. 이미 시작한 run은 Pause·재개 시에도 당시 설정을 유지한다.
- **수집 시작**은 저장한 목표 N episodes를 실행하고, **한 번만 실행**은 저장된 목표를 바꾸지 않고 **1 episode와 reset**을 실행한다. 두 버튼 모두 완료 후 다음 run을 자동 시작하지 않는다. 수집 화면에서 현재 run과 다음 run의 목표를 따로 확인할 수 있다. 파이프라인 rollout은 파이프라인의 목표를 따른다.
- 추론 수준 **Auto**는 각 episode의 task/reset을 `medium`으로 시작한다. GPT가 정밀 조작, 접촉·상태의 불확실성, 실패 복구 등을 판단해 `xhigh`를 요청하면 **동작 없이 새 관측으로 재판단**한다. 안정된 상태에서 다음 동작이 단순하다고 판단하면 다음 요청을 `medium`으로 돌린다. 전환 기준은 특정 task·물체에 종속되지 않으며, 전환 요청도 GPT 호출 수·비용 한도에 포함된다. 재판단을 마치면 현재 run의 제어 흐름을 계속 따르며, 한 번만 실행도 episode와 reset 전체를 수행한다. 학습 파이프라인의 관측 전용 성공 판정기는 Auto 설정에서 `xhigh`를 사용한다.
- 데이터 화면은 성공 episode를 **Optimal / Sub-optimal / 분류 정보 없음**으로 구분한다. 매 action마다 GPT에 실패 여부를 묻는 대신 **실제로 적용된 gripper 닫기 시도**를 센다. 닫기 유지 명령의 반복은 중복 계산하지 않으며, 다시 연 뒤 닫는 시도는 추가한다. Task와 Reset의 횟수는 분리하며 Task 횟수만 예상 기준과 비교한다. 현재 분류는 예상 횟수와 일치하면 첫 범주, 초과하면 추가 grasp, 더 적거나 계수·기준이 불확실하면 분류 정보 없음이다. 최적 경로나 실제 무실패를 보장하는 기준은 아니며, 최종 성공 여부는 환경의 독립 verifier가 판정한다. 기존 GPT 실패 보고 기반 성공 기록은 원래 근거를 유지하면서 실패 보고가 없으면 Optimal, 복구 보고가 있으면 Sub-optimal로 표시한다. 새 횟수 기준으로 소급 해석하지 않는다. 프로젝트의 episode 표시 번호는 여러 run을 통틀어 시간순으로 유지하며, 저장 경로와 run 내부 번호는 변경하지 않는다.
- **설정 → Task grasp 기준**에서 예상 닫기 횟수와 GPT 추정 근거를 확인하고 직접 수정한 뒤, 하단 **설정 저장**으로 다른 설정과 함께 저장할 수 있다. 횟수를 바꾸지 않은 저장은 기존 GPT 추정 기준을 유지한다. 자동 기준이 없으면 명시적인 수집 시작 후 GPT가 관측·task·action 규약을 바탕으로 한 번 추정하며, 같은 task의 기준은 캐시해 재사용한다. 이 관측 전용 추정 요청의 사용량·비용도 별도로 기록한다. 수동 기준을 비워 저장하면 다음 수집 시작 시 자동 추정한다. 기준 변경은 새 run부터 적용하며 기존 episode는 당시 횟수·기준 버전을 유지한다.
- 데이터 목록 오른쪽 위의 **선택 보관 / 선택 삭제**를 누르면 체크박스가 나타난다. 현재 필터의 관리 가능한 데이터를 선택한 뒤 **일괄 보관 / 일괄 삭제**로 처리하며, 삭제는 선택 목록을 다시 확인한다. 보관함에서는 일괄 보관 해제도 가능하다. 일부 요청이 실패하면 해당 항목과 이유를 남기며, 연결 오류가 발생하면 나머지 요청을 보내지 않는다. BC 평가·RL 학습·RL 평가 데이터는 기존 데이터도 **Rollout 01** 형식으로 표시하고, 수집·외부 시연은 **Episode** 표기를 유지한다. 프로젝트 내 기존 표시 번호와 저장 파일은 유지한다.
- 데이터의 각 episode에서 **보관**하면 기본 목록·demo 집계에서 빠지고 **보관함**에서 확인·복원할 수 있다. **삭제**는 확인 후 해당 episode의 관측·action 기록과 연결 영상을 영구 삭제한다. 대시보드의 데이터 사용 비용과 토큰 사용량은 사용 중·보관 중 episode만 합산하며, 삭제한 episode는 제외한다. 실행 중인 run의 데이터는 보관·삭제할 수 없다.
- 수집 화면의 실행 이벤트에는 episode 시작·결과·reset·Pause 같은 진행 상황을 표시한다. GPT가 반환한 동작 설명, action과 실행 결과는 **데이터 → episode 상세**에서 확인한다. 내부 추론 원문을 저장하거나 표시하는 기능은 아니다.

- **Pause / Resume**: 현재 한 control tick이 끝나면 멈춘다. 초기화/reset 중이면 해당 작업이 끝난 뒤 멈춘다. 대기 중인 GPT 요청과 미실행 action chunk를 폐기하고, 재개 시 새 관측으로 판단한다.
- **한 번만 실행 / Run once**: GPT 판단 1회가 아니라, 새 run에서 **1 episode와 reset 전체**를 실행한다. 진행 중이거나 일시정지된 run은 **수집 재개**로 이어간다.
- **수집 중지 / Stop**: 현재 episode와 실행을 중지한다. 상세 콘솔의 **Reset**은 기존 실행을 종료하고 같은 설정으로 새 run을 만드는 수동 초기화 기능이다.
- 기본 `GPT arm reset`은 성공 후에도 같은 물리 상태에서 GPT가 팔을 조작해 복구한다. **마지막 episode에도 reset을 수행한다.** Reset 실패·예산 소진 시 정지하며 자동으로 환경 reset하지 않는다.
- `free_space`(**대상 복구 + 안전 후퇴 후 환경 초기화**)는 시뮬레이터 전용 선택 모드다. GPT가 대상을 복구하고 gripper를 해제한 뒤 안전 영역으로 후퇴한다. 환경의 독립 verifier가 대상·주변 물체 복구, 해제, 충돌 여유와 정지 상태를 검증한 후에만 simulator reset을 수행한다. **마지막 episode도 이 검증과 환경 초기화까지 마친다.** 검증 실패나 지원하지 않는 환경에서는 자동 초기화를 수행하지 않는다.
- 기본 task는 `libero_90`의 **7번, 상단 서랍 열기**다. 주변 간섭이 적은 배치를 사용한다. `libero_goal` 0번 middle drawer와 `libero_90` 11번 top drawer도 독립 reset verifier를 지원한다. Reset은 서랍 닫기·gripper 열기·초기 팔 pose 근처 복귀·주변 물체 보존을 검사하며 초기 상태와 bit 단위로 동일하지는 않다.
- `episodes > 1`이면 검증된 GPT reset 후 다음 demo를 수집한다. `Environment reset`은 명시적으로 선택하는 별도 모드다. **Hard reset environment**는 현재 장면을 폐기하는 수동 복구 기능이다.
- 카메라, 실제 관측 시각, task, step/판단 수, action·실행 receipt·이벤트를 UI에서 확인할 수 있다. 실시간 화면은 실제 카메라만 표시하고 reset 목표 참고 이미지는 숨긴다. 모델이 사용하는 내부 참고 이미지와 reset 검증 기준은 유지된다.

키·시뮬레이터 없이 UI/제어 흐름만 확인하려면 `rilo --demo`를 사용한다. **명시적인 2D 테스트 환경과 scripted policy이며 LIBERO/GPT를 사용하지 않는다.**

## 설정

`.env.example` 참고. 주요 변수는 `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`, `RILO_ENV`, `RILO_POLICY`, `LIBERO_SUITE`다. `OPENAI_BASE_URL`은 Responses·vision·function calling을 지원하는 호환 endpoint에만 사용한다.

`.env`의 모델·추론 수준은 프로젝트에서 별도로 지정하지 않았을 때 사용하는 서버 기본값이다. UI에서는 지원 모델별 추론 수준만 선택할 수 있고, 실행 manifest에 실제 적용한 모델·추론 수준을 기록한다. API key는 기존처럼 서버 환경 변수에서만 읽는다.

`RILO_PROMPT_FILE=prompts/control.md`를 설정하면 다음 GPT 판단부터 파일 수정을 읽으므로 물리 환경을 유지하며 프롬프트를 개선할 수 있다. 호출마다 정확한 프롬프트 hash와 snapshot을 기록한다. 실험 성공률 비교 시에는 프롬프트와 reasoning effort를 고정한다.

LIBERO는 `robosuite==1.4.1`, `mujoco==3.2.7`과 고정 upstream commit을 사용한다. 설치 스크립트가 `vendor/LIBERO`와 `artifacts/libero-config`를 만들며 기존 `~/.libero`를 변경하지 않는다.

렌더링은 기본 `MUJOCO_GL=osmesa`(Linux의 `libOSMesa` 필요). NVIDIA 드라이버가 정상인 머신은 `MUJOCO_GL=egl`을 사용할 수 있다. 첫 초기화에는 shader/JIT 준비 시간이 든다. `rilo doctor`로 설치 상태를 확인한다.

서버는 기본 localhost에만 열린다. 원격 확인은 `ssh -L 8000:127.0.0.1:8000 user@robot-host`로 터널링한다. 이 프로토타입에는 사용자 인증이 없으므로 공개 인터넷에 노출하지 않는다. **프로세스/worker는 하나만 실행한다.**

## 구조와 확장

| 위치 | 책임 |
|---|---|
| `src/rilo/contracts.py` | `Observation`, `ActionSpec`, `Decision`, adapter/policy 계약 |
| `src/rilo/adapters/libero.py` | LIBERO 관측·초기화·action 해석·성공 predicate |
| `src/rilo/policies/openai_policy.py` | 이미지·상태·최근 receipt → 검증된 수치 action |
| `src/rilo/runner.py` | 제어권, Pause/Resume, 실행 예산, episode 전환 |
| `src/rilo/recording.py` | control tick별 관측과 실행 기록 |
| `src/rilo/server.py`, `src/rilo/web/` | 환경에 독립적인 HTTP API와 웹 콘솔 |
| `src/rilo/learning/` | dataset, Diffusion Policy, progress, residual TD3·replay |
| `src/rilo/pipeline.py` | 순차 수집·학습·평가, 실행 권한과 모델 레지스트리 |

새 환경은 `action_spec`, `describe`, `reset`, `observe`, `step`, `pause`, `close`를 구현하고 `RILO_ENV=my_package.adapter:create_adapter`로 지정한다. `list_tasks`는 선택 사항이다. Factory는 인자 없이 adapter를 반환한다. Action 차원·단위·좌표계·gripper 의미는 **adapter가 선언**하며 runner/policy에 LIBERO의 7차원 가정이 없다. 새 정책도 `decide`와 `close`를 구현해 `RILO_POLICY=module:factory`로 연결할 수 있다.

GPT reset을 지원하는 환경은 `begin_reset`과 `begin_task`도 구현한다. 두 메서드는 물리 상태를 유지하며 목표만 바꾸고, `Observation.success`는 현재 task/reset 목표의 독립 검증 결과다. `free_space`는 시뮬레이션 여부·지원 정보와 독립 검증을 제공하는 `begin_free_space_reset`이 추가로 필요하다. 프로젝트 환경 연결 API는 관측만 준비하고, 명시적인 start/step에서 run 설정을 확정한다. 검증된 복구 상태 또는 초기화가 완료된 장면에서는 환경을 다시 연결하지 않고 다음 run을 시작할 수 있다. 안전하게 이어갈 수 없는 중지·오류 상태는 명시적인 **환경 다시 연결**로 장면을 초기화한다.

모든 simulator 접근은 같은 전용 thread에서 이루어진다. `step`은 한 번의 제한된 control tick, `observe`는 동작 없는 새 관측이다. 실제 로봇 adapter는 `pause`를 로컬 정지에 연결하고 controller watchdog을 별도로 구현해야 한다. 웹 Pause는 물리 E-stop을 대체하지 않는다.

LIBERO action은 **정규화된 OSC delta**이며 직접적인 metre 명령이 아니다. Translation/rotation scale과 repeat 의미를 adapter가 GPT에 전달한다. Object ground-truth pose는 GPT 입력에 포함하지 않고, 성공은 LIBERO predicate로 확인한다. GPT의 성공 주장만으로 성공 데이터를 만들지 않는다.

한 GPT 응답은 최대 8개 action vector를 담으며, 각 vector는 최대 10 tick 반복하고 전체 chunk는 최대 30 tick이다. 현재 LIBERO는 20 Hz이므로 tick 하나는 시뮬레이션 0.05초다. GPT 응답 대기 중에는 `env.step()`을 호출하지 않아 시뮬레이션 시간이 흐르지 않는다. 실제 로봇의 대기 중 자세 유지와 상태 변화 처리는 별도 로컬 controller의 책임이다.

## 기록과 검증

`artifacts/runs/<run_id>/`에 실행 설정, 환경 설명, `episode-*/transitions.jsonl`, 카메라 JPEG, episode summary를 저장한다. 요청 action과 실제 적용 action, 관측 ID·시간, 부분 실행 receipt를 남긴다. Reset 구간은 task transition과 분리한다. History는 최근 실행 receipt로 제한하며, 모델에는 현재 카메라 이미지와 측정 상태를 함께 보낸다.

**실제 실행 데이터는 Git에 올리지 않는다.** `.env`와 그 변형, `artifacts/` 전체(run·이미지·영상·비용 원장·프로젝트 상태·학습 checkpoint), 가상환경과 `vendor/`는 `.gitignore`로 제외한다. 저장소에는 코드·테스트·설정 예시만 포함하며, 기존 로컬 데이터는 그대로 유지된다.

```bash
python -m pytest -q
```

테스트는 API 호출 없이 action 검증, 잘못된 성공 주장, 제어 예산, 기록, Pause 중단·늦은 응답 폐기·재개, GPT reset 순환, 비용 계산, 웹 API를 확인한다.

각 run의 `api_calls.jsonl`에는 유효하지 않은 응답까지 포함한 API 호출별 raw usage와 추정 비용을 기록한다. Input/cache read/cache write/output을 구분하고 reasoning token은 output의 부분집합으로 표시한다. 응답을 받지 못한 timeout·취소 등은 비용 미확인으로 남긴다. UI 달러 예산은 응답 수신 뒤 확인하므로 마지막 한 요청 비용만큼 초과할 수 있다. 비용은 [공식 단가](https://developers.openai.com/api/docs/models/gpt-6-astra) 기준 추정이며 실제 청구서와 다를 수 있다.

`python scripts/report_costs.py artifacts/runs/<run_id> --json artifacts/costs.json --markdown artifacts/costs.md`로 task/reset 평균과 실패 비용까지 포함한 성공 demo당 비용을 다시 계산한다. 여러 run 경로를 전달하면 모델·reasoning effort·코드 버전별로 구분한다.

## 외부 LIBERO 시연 가져오기

[LIBERO 공식 데이터 저장소](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)에서 프로젝트의 suite와 정확한 BDDL task 이름에 맞는 HDF5 파일을 사용한다. 서버를 중지한 상태에서 다음 명령으로 변환·연결할 수 있다. `--commit-offline`을 생략하면 프로젝트에 노출되지 않는 staging 폴더까지만 만든다.

```bash
python scripts/import_libero.py /path/to/task_demo.hdf5 \
  --data-dir artifacts/runs --project-id <PROJECT_ID> \
  --task-name <CANONICAL_BDDL_TASK_NAME> --published-successful --commit-offline
```

가져온 episode는 회색 **외부 소스**로 표시하며 원본 파일 SHA-256, task와 demo ID를 보존한다. 같은 시연을 다시 가져와도 중복 연결하지 않는다. 공식 파일은 action 이후 관측을 저장하므로 입력 관측이 없는 첫 action만 제외하고 실제 마지막 성공 관측까지 정렬한다. 카메라는 upright RGB로 변환한다. 기존 LIBERO 기록과의 혼합 학습은 공통 측정값인 EEF 위치·quaternion, 관절 위치, gripper 위치를 사용한다. 원본 기록의 추가 측정값은 유지하며 외부 데이터에 없는 속도·reset·GPT 비용·Optimal 분류를 만들어 넣지 않는다.

## 학습 파이프라인

`pip install -e '.[learning,dev]'` 후 프로젝트 **모델** 탭에서 학습 조건을, **평가** 탭에서 평가 조건을 저장한다. 전체 파이프라인은 **개요 → Orchestration → 전체 파이프라인 실행** 팝업에서 주요 조건을 선택해 시작한다. 개별 학습은 **모델**, 개별 평가는 **평가** 탭에서 실행한다.

**GPT 수집 → BC 학습 → BC 평가 → 목표 미달 시 추가 수집 → progress 학습 → residual TD3 / 평가 반복**. 수집 목표는 보관·삭제 데이터를 제외한 사용 중 성공 task episode의 누적 수이며 수동 수집 결과도 포함한다. 개요의 수집 카드에는 이 성공 수와 목표가 표시된다. 자동 실행은 부족한 성공 episode만 추가 수집하고, 현재 데이터·설정에 맞는 기존 모델·평가 결과를 재사용해 남은 단계부터 이어간다. 모든 목표를 충족했다면 추가 실행을 시작하지 않는다. BC 평가가 목표에 미달하면 수집 목표를 늘려 추가 데이터를 모으고 다시 학습한다. 개요에는 03 평가에서 01 수집으로 돌아가는 화살표가 표시되며, Progress 모델은 개별 모델 학습에서 계속 관리한다. 수집과 supervised 학습은 순차 실행한다. 성공·실패를 모두 기록하고, DP·progress는 기본적으로 성공 task 구간만, RL replay는 두 종류를 모두 사용한다. 기본 agent 모드는 GPT action으로만 reset한다. 프로젝트에서 `free_space`를 선택하면 독립 검증을 통과한 안전 후퇴 뒤 simulator reset을 수행한다.

DP는 ResNet18 기반 약 13.7M 모델, residual은 frozen DP 위의 작은 MLP다. 모델·지표·설정은 `artifacts/learning/<project_id>/`에 저장한다. **실제 데이터 학습과 성능 측정은 아직 실행하지 않았다.** 합성 fixture로 전체 6단계 smoke test를 제공한다.

[모델 구조·실행 방법·설계 차이](docs/learning.md) · [기본 hyperparameter](configs/pipeline.json). CLI는 `python scripts/run_pipeline.py --project <ID> status`로 상태를 확인하고 `start --stage bc`, `start --stage rl`, `start --auto`로 명시적으로 실행한다.

**현재 범위:** GPT 직접 제어, LIBERO, 데이터 관리, DP·progress·residual RL과 순차 orchestration/UI. 실제 로봇 adapter와 학습 성능 평가는 후속 검증이 필요하다.

## 실제 측정 · 2026-10-27

`gpt-6-astra / xhigh`, `libero_90 / task 7`에서 같은 초기 배치로 연속 3회 수행했다. Task와 GPT reset 모두 **3/3 성공**, 사이클 사이 환경 reset **0회**였다. 3회 소표본이며 다른 배치/seed로 일반화된 성공률은 아니다.

Demo 1개당 평균 **task 129,564 tokens / $1.62**, **reset 112,794 tokens / $1.28**, **합계 242,358 tokens / $2.90 / 약 7.8분**. 3회 합계 $8.70, 실패·디버깅·task 선택 실험까지 포함한 전체 추정 비용은 **$17.95**였다. 초기 middle-drawer 장면은 접시 복구에 실패하여 최종 task에서 제외했다.

측정 원본은 로컬의 `artifacts/final-costs.md`, `artifacts/all-experiments-costs.md`, `artifacts/collection-campaign.json`에 보관하며 GitHub에는 포함하지 않는다.

저장된 5개 episode 영상은 `artifacts/videos/index.html`에서 확인한다. 외부/손목 카메라와 TASK/RESET 표시가 포함되며 **API 대기 시간을 제외한 시뮬레이션 시간 기준 20 fps**다.

```bash
python scripts/serve_videos.py
# http://127.0.0.1:8002
```

다른 기록은 `python scripts/export_videos.py --help`의 옵션으로 MP4로 변환할 수 있다. 이 변환은 저장된 이미지만 사용하며 로봇 실행이나 API 호출을 하지 않는다.
