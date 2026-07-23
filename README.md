# Agentic AI 실습 자료 — 자율 에이전트 개발

LangChain·LangGraph를 이용해 AI App → Workflow → Agent로 이어지는 개념을 단계별로
익히는 Jupyter Notebook 실습 저장소입니다. 

## 디렉터리 구조

```text
agentic-ai-lab/
├─ README.md
├─ 00_environment_check.ipynb        # 모든 Track 공통, 최초 1회 환경 점검
├─ .env.example
├─ pyproject.toml / uv.lock          # uv 기반 의존성 관리
├─ src/agentic_ai/                   # 공통 Python 패키지
│  ├─ config.py
│  ├─ paths.py
│  ├─ models.py
│  ├─ schemas.py
│  ├─ tools.py
│  ├─ states.py
│  ├─ logging_utils.py
│  ├─ retrieval_utils.py
│  └─ notebook_utils.py
├─ track1_core/                      # Track 1: 핵심 개념 (01-1 ~ 06)
│  ├─ 01-1_ai_app_workflow_agent.ipynb ... 05-1_integrated_knowledge_agent.ipynb
│  ├─ 06_my_agentic_ai_worksheet.ipynb   # 자유 설계 캡스톤 워크시트
│  ├─ solution/                      # 강사용 정답 Notebook (01-1 ~ 05)
│  └─ worksheet/                     # 학생용 TODO Notebook (01-1 ~ 05)
├─ track2_practical1/                # Track 2: 실전 응용 1 — 사내 보고서(work) 문서
│  ├─ 1_work_document_chunking.ipynb
│  ├─ 2_work_embedding_vector_retrieval.ipynb
│  ├─ 3_work_agentic_rag_basic.ipynb
│  └─ 4_work_integrated_knowledge_agent.ipynb
├─ track3_practical2/                # Track 3: 실전 응용 2 — 법령(HWPX) 문서
│  ├─ 1_aitrust_document_chunking.ipynb
│  ├─ 2_aitrust_embedding_vector_retrieval.ipynb
│  ├─ 3_aitrust_agentic_rag_basic.ipynb
│  └─ 4_aitrust_integrated_knowledge_agent.ipynb
├─ data/                             # 실습용 원본 데이터
│  ├─ sample_inputs.jsonl
│  ├─ sample_report.txt / sample_report5.docx
│  └─ 인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 (doc/hwpx/pdf)
└─ outputs/                          # 실행 결과 (실습 시 자동 생성)
   ├─ logs/
   ├─ retrieval/
   ├─ vectorstore/chroma/
   ├─ checkpoints/
   └─ evaluation/
```

## 환경 설정

모든 Track과 `00_environment_check.ipynb`는 저장소 루트의 `pyproject.toml`,
`uv.lock`, `.venv` 하나를 공유합니다. Notebook마다 별도 가상환경을 만들지 않습니다.

1. Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)를 준비합니다.
2. 저장소 루트에서 의존성을 동기화합니다.

   ```bash
   uv sync
   ```

   `pyproject.toml`이 `src/agentic_ai`를 프로젝트 패키지로 설치하므로 Notebook에서
   `sys.path`를 직접 수정할 필요가 없습니다. 패키지를 추가할 때도 저장소 루트에서
   `uv add <패키지명>`을 사용합니다.

3. `.env.example`을 복사해 `.env`를 만들고 API Key를 채웁니다.

   ```bash
   cp .env.example .env
   ```

   ```env
   OPENAI_API_KEY=your_api_key_here
   CHAT_MODEL=gpt-4.1-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

   `.env` 파일은 절대 커밋하지 않습니다. 모델명은 코드 셀에 직접 작성하지 않고
   `src/agentic_ai/models.py`의 `get_chat_model()` / `get_embedding_model()`을 통해서만
   사용합니다.

4. Jupyter 또는 IDE에서 루트 `.venv`의 Python을 Kernel로 선택합니다.
   Windows 기준 Kernel 경로는 `.venv\Scripts\python.exe`입니다.

5. `00_environment_check.ipynb`를 열어 Python, 패키지, API Key, Chat Model,
   Embedding Model, Chroma 점검을 모두 통과합니다.

## 실습 구성

### Track 1 — `track1_core/` (핵심 개념)

AI App/Workflow/Agent 비교부터 LangGraph 기반 Tool Agent, Agentic RAG,
Memory/Human-in-the-Loop, 통합 Agent까지 순서대로 다루는 기본 커리큘럼입니다.

| 번호 | 제목 | 핵심 내용 |
|---|---|---|
| 01-1 | AI App vs Workflow vs Agent | 동일한 문제를 세 가지 구조로 구현하며 차이를 관찰 |
| 01-2 | LLM Message & Structured Output | System/User Message, Pydantic 기반 구조화 출력과 검증 |
| 01-3 | Tool Calling 기초 | Tool Schema 작성, `bind_tools`, Tool Call 실행, 오류 처리 |
| 01-4 | Agent Loop와 ReAct | Plan-Act-Observe-Evaluate 반복 구조, 종료 조건과 최대 반복 안전장치 |
| 02-1 | LangGraph State, Node, Edge | `StateGraph`/`START`/`END`, Node의 부분 State 갱신 |
| 02-2 | 조건 분기와 Multi-Route Workflow | 규칙 기반 Router와 Conditional Edge |
| 02-3 | MessagesState와 ToolNode로 만드는 Tool Agent | `MessagesState`, `ToolNode`, `tools_condition`, 복수 Tool Call |
| 03-1 | 문서 전처리와 Chunking | 고정 길이·Overlap·구조 기반 비교, `RecursiveCharacterTextSplitter` |
| 03-2 | Embedding과 Vector Retrieval | 디스크 영구 Chroma DB, Top-k 검색, Metadata Filter |
| 03-3 | Agentic RAG 기초 | 검색 판단 → 검색 → 답변 Graph, Query Rewrite·재검색 종료 조건 |
| 04-1 | Context, State, Memory | `InMemorySaver`+`thread_id` 단기 기억, `InMemoryStore`+namespace 장기 기억 |
| 04-2 | Checkpoint와 Human-in-the-Loop | `interrupt()`/`Command(resume=...)`, 재개 시 부작용 중복 문제와 멱등성 |
| 05-1 | 통합 Knowledge Agent | 10-Node Adaptive Workflow에 Tool Agent Loop 통합, 실제 HITL |
| 06 | 나만의 Agentic AI 설계·구현 워크시트 | 지금까지 배운 State/Memory/RAG/HITL을 자유 주제로 통합 설계·구현하는 캡스톤 실습 |

- `track1_core/*.ipynb`(01-1 ~ 05-1): 완성된 예제/설명이 포함된 강의용 버전
- `track1_core/solution/`: 01-1 ~ 05 강사용 정답 Notebook
- `track1_core/worksheet/`: 01-1 ~ 05 학생용 TODO Notebook (빈칸을 채우며 학습)
- `track1_core/06_my_agentic_ai_worksheet.ipynb`: 정답이 없는 자유 설계 캡스톤. 별도 설계
  문서(`Part4_설계_워크시트.docx`, `textbook/` 자료 참고)에서 요구사항과 아키텍처를 먼저
  정한 뒤, State → Memory → RAG(선택) → Node → Routing → Graph → 시나리오 테스트 순서로
  TODO를 채운다.

### Track 2 — `track2_practical1/` (실전 응용: 사내 보고서)

Track 1의 03-1 ~ 05 개념(Chunking → Retrieval → Agentic RAG → 통합 Agent)을
`data/sample_report5.docx`(사내 리모트워크 운영 보고서) 같은 실제 업무 문서에
적용하는 실습입니다. 모든 코드가 완성되어 있어 직접 실행하며 결과를 관찰하는
데 초점을 둡니다.

### Track 3 — `track3_practical2/` (실전 응용: 법령 문서)

같은 4단계 파이프라인을 `data/인공지능 발전과 신뢰 기반 조성 등에 관한 기본법`
HWPX 법령 문서에 적용합니다. 조·항·호·목 번호 체계를 인식해야 하는 법령 문서
특유의 Chunking·검색 이슈(번호 충돌, 짧은 조문의 문맥 손실 등)를 다룹니다.

### `textbook/` (강사용 교재 제작 자료)

`python-docx` 기반으로 Notebook 내용을 Word 교재(챕터별 `build_doc_ch*.py`)와
Mermaid 다이어그램(`make_diagrams_*.py`)으로 변환하는 스크립트 모음입니다.
`_template/`에는 문서 서식 템플릿이, `_to_delete/`에는 정리 대상 초안이 들어
있습니다. Notebook 실습 자체를 실행하는 데는 필요하지 않습니다.

## Notebook 실행 방법

1. `00_environment_check.ipynb`를 먼저 실행한다.
2. 원하는 Track의 Notebook을 번호 순서대로 실행한다.
   - Track 1은 개념 학습(`track1_core/`, 필요 시 `worksheet/`로 직접 풀이),
   - Track 2·3은 같은 개념을 다른 문서 도메인에 적용하는 응용 실습이다.
3. 각 Notebook의 환경 설정 셀은 설치된 `agentic_ai` 패키지에서 설정·경로·모델을 가져온다.
4. 실행 결과 로그는 `outputs/logs/`에, 검색 결과는 `outputs/retrieval/`에,
   Chroma 데이터는 `outputs/vectorstore/chroma/`에, Human-in-the-Loop 체크포인트는
   `outputs/checkpoints/`에 저장된다.

## 공통 Python 모듈 (`src/agentic_ai/`)

- `config.py`: `.env`에서 API Key와 모델명을 읽어 `Settings`를 제공합니다.
- `paths.py`: `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, `CHROMA_DIR`와 경로 Helper를 제공합니다.
- `models.py`: `get_chat_model()` / `get_embedding_model()` Factory를 제공합니다.
- `schemas.py`: Structured Output 검증에 사용하는 `InputAnalysis` Pydantic 모델을 정의합니다.
- `tools.py`: `calculate()`(AST 기반 안전한 사칙연산, `eval()` 미사용), `search_keyword()`,
  `load_document()`, `check_required_fields()` 등 Tool Calling 실습용 함수를 제공합니다.
- `states.py`: LangGraph 실습에서 공유하는 `AgentState` TypedDict를 정의합니다.
  Notebook별로 필요한 추가 State(`RouterState`, `RAGState` 등)는 각 Notebook 안에서
  직접 정의합니다.
- `logging_utils.py`: 실행 상태 출력(`print_state`, `print_steps`)과 로그 저장/불러오기
  (`save_log`, `save_csv`, `create_run_log`, `load_jsonl`) 기능을 제공합니다.
- `retrieval_utils.py`: 디스크 기반 Chroma Collection 생성(`get_chroma_store`)과
  기존 문서 재사용(`add_documents_if_empty`)을 담당합니다.
- `notebook_utils.py`: 공통 환경 요약 출력(`print_environment_summary`)과 LangGraph
  시각화(`show_graph`)를 제공합니다.

## 데이터와 출력

- `data/`: `sample_inputs.jsonl`, `sample_report.txt`, `sample_report5.docx`(Track 2용
  사내 보고서), 그리고 Track 3용 `인공지능 발전과 신뢰 기반 조성 등에 관한 기본법`
  원문(doc/hwpx/pdf)이 들어 있습니다.
- `outputs/`: Notebook 실행 시 자동 생성되는 로그(`logs/`), 검색 결과(`retrieval/`),
  Chroma 벡터스토어(`vectorstore/chroma/`), HITL 체크포인트(`checkpoints/`), 평가
  결과(`evaluation/`)를 담습니다.

## 참고

- `main.py`는 uv 프로젝트 스캐폴딩이 만든 자리표시자 스크립트로, 실습 흐름과는 무관합니다.
- `requirements.txt`는 참고용이며, 실제 의존성 관리는 `pyproject.toml` + `uv.lock`(uv)이 기준입니다.
# agentic_ai_lab_202607
