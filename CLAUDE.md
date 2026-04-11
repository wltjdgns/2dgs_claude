# 2DGS-SAM 프로젝트 버전관리 규칙

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 확인된 안정 상태만 (train + render 결과 시각적 검증 완료 후 merge) |
| `feat/<실험명>` | 코드 수정 및 실험 (예: `feat/specular-normal-fallback`) |

## 태그 전략

형식: `v<major>.<minor>-<설명>`

| 태그 | 커밋 | 설명 |
|------|------|------|
| `v1.0-baseline` | f0e4213 | specular planar 수정 전 안정 기준점 |
| `v2.0-specular-fix` | 90f3f89 | normal fallback 추가 (태블릿 탐지) |
| `v2.1-threshold-12deg` | d4d8de0 | fallback threshold 과탐지 억제 |

## 작업 흐름

### 새 실험 시작
```bash
git checkout -b feat/<실험명>
```

### 실험 완료 후 결과 좋음 → main merge + 태그
```bash
git checkout main
git merge feat/<실험명>
git tag v<X.Y>-<설명>
git push origin main --tags
git branch -d feat/<실험명>
```

### 결과 나쁨 → 브랜치 삭제, main 무변경
```bash
git checkout main
git branch -d feat/<실험명>
```

### 특정 버전으로 롤백
```bash
# 파일 하나만 롤백
git checkout <tag> -- utils/planar_utils.py

# 전체 롤백
git checkout <tag>
```
