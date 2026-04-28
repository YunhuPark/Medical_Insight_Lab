# 한국의료정보학회지 투고 체크리스트

투고 링크: https://www.e-jhis.org

---

## 제출 전 준비 사항

### 1단계 — 파일 변환 (지금 해야 할 것)

- [ ] `paper_draft.md` → **HWP 또는 DOCX** 변환
  - 한글(HWP) 권장 (국내 학술지 표준)
  - MS Word DOCX도 대부분 허용
  - 변환 방법: VS Code에서 Markdown → DOCX (Pandoc 활용)
    ```bash
    pandoc paper_draft.md -o paper_draft.docx
    ```
  - 또는 구글 독스에 붙여넣기 후 DOCX 다운로드

- [ ] 표 1을 본문 내 삽입 (Markdown 표 → Word 표로 변환 확인)
- [ ] 그림 파일 준비 (results/ 폴더의 주요 PNG 파일)
  - 필수: `v2_04_psm_analysis.png` (PSM 결과)
  - 필수: `v2_11_type_fear_interaction.png` (채널 유형별 효과)
  - 권장: `v2_05_covid_experiment.png` (DiD)
  - 권장: `v2_13_sci_statistical_report.png` (SCI 통계 보고)

---

### 2단계 — 학회 홈페이지 확인 사항

- [ ] 투고 안내 페이지 접속: https://www.e-jhis.org/authors/authors.php
- [ ] 원고 분량 제한 확인 (보통 A4 20페이지 이내)
- [ ] 참고문헌 형식 확인 (Vancouver 방식 vs APA)
- [ ] 그림/표 형식 요구사항 확인 (해상도 300 DPI 이상)
- [ ] 투고비 확인 (학회마다 상이, 보통 없거나 소액)

---

### 3단계 — 제출 파일 목록

| 파일 | 내용 | 상태 |
|------|------|------|
| `paper_draft.docx` | 논문 원고 (변환 필요) | ⬜ 변환 필요 |
| `cover_letter.md` | 커버레터 | ✅ 완성 |
| 그림 파일 4개 | 분석 결과 이미지 | ✅ results/ 폴더에 있음 |
| 저자 정보 | 온라인 투고 시스템에서 입력 | ⬜ 시스템 접속 후 |

---

### 4단계 — 온라인 투고 시스템 입력 정보

투고 시스템에서 직접 입력할 내용:

```
논문 제목 (국문):
한국 의료 유튜브에서 공포 소구가 조회수에 미치는 인과적 영향:
성향점수매칭과 이중차분법을 활용한 분석

논문 제목 (영문):
Causal Effects of Fear Appeals on View Counts in Korean Medical YouTube:
A Propensity Score Matching and Difference-in-Differences Approach

저자명: 박윤후
소속: 인제대학교 의료IT학과
이메일: byunhu35@gmail.com

주요어 (국문): 공포 소구, 의료 유튜브, 인과 추론, 성향점수매칭, 이중차분법, 공중보건 커뮤니케이션
Keywords: fear appeal, medical YouTube, causal inference, propensity score matching, difference-in-differences, public health communication

연구비 지원: 없음
이해충돌: 없음
```

---

## 투고 후 예상 일정

| 단계 | 소요 기간 |
|------|----------|
| 투고 접수 확인 | 1~3일 |
| 편집위원 초심 (게재 가능 여부) | 1~2주 |
| 동료 심사 (peer review) | 4~8주 |
| 심사 결과 통보 | 게재 / 수정 후 게재 / 게재 불가 |
| 수정 요청 시 재투고 기간 | 4주 이내 |
| 최종 게재 확정 후 온라인 출판 | 1~4주 |

**총 예상 기간: 2~4개월**

---

## 리젝트 대비 플랜

| 심사 결과 | 대응 |
|----------|------|
| 수정 후 게재 | 심사의견 반영 후 재투고 (내가 답변서 초안 도와드림) |
| 게재 불가 | 수정 보완 후 → 정보과학회논문지 또는 한국정보통신학회논문지 재투고 |
| 반복 리젝트 | JMIR (영문, 국제 SCI) 또는 학회 발표(AMIA Abstract)로 전환 |

---

## 지금 당장 해야 할 것 (우선순위)

1. **paper_draft.md → DOCX 변환** (Pandoc 또는 구글 독스)
2. **https://www.e-jhis.org 접속** → 투고 안내 확인
3. **온라인 회원가입** (투고 시스템 계정 필요)
4. **커버레터 DOCX 변환**
5. **온라인 투고 시스템에서 파일 업로드**
