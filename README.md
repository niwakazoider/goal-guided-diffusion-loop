# Goal-Driven Stable Diffusion Prompt Optimization Loop

自然言語の目標、または参照画像から生成した目標をもとに、Gemini とローカル Stable Diffusion を反復実行し、画像を目標へ近づける Python スクリプトです。

`GOAL.jpg` がある場合は、Gemini が画像を日本語の `TARGET_GOAL` へ変換し、生成画像を次の2つの観点から評価します。

- テキスト化された目標仕様への適合
- 参照画像との構造的な類似

参照画像との比較では、構図、人物配置、ポーズ、シルエット、物体の大きさ・方向、背景形状だけを評価します。色、線の質、塗り方、描画密度、完成度などの STYLE は参照画像から引き継がず、テキスト化された目標を優先します。

## 主な機能

- `GOAL.jpg` から `TARGET_GOAL` を自動生成
- `GOAL.jpg` がない場合は `GOAL.txt` を使用
- GoalCompiler による動的な評価基準の生成
- Stable Diffusion 用 Positive / Negative Prompt の自動生成
- Gemini Planner によるPromptの反復改善
- AUTOMATIC1111 によるローカル画像生成
- Gemini Evaluator による画像評価
- 人間による追加指示と手動合格

## 処理の流れ

```text
GOAL.jpg がある場合
    ↓
Gemini が画像を TARGET_GOAL へ変換
    ↓
GoalCompiler が criteria と初期Promptを生成
    ↓
Gemini Planner
    ↓
Stable Diffusion WebUI API
    ↓
生成画像
    ├─ TARGET_GOAL / criteria への適合評価
    └─ GOAL.jpg との構造比較
    ↓
未達項目とPrompt修正候補をPlannerへ返す
    ↓
合格または最大反復数まで繰り返す
```

`GOAL.txt` だけを使用する場合は、参照画像との構造比較を行わず、テキスト目標への適合だけを評価します。

## 必要環境

- Python 3.10以上
- Google Gemini APIキー
- AUTOMATIC1111
- Stable Diffusion WebUI API

想定モデルは `illustrious_pencil-XL` ですが、実際にはWebUIで現在ロードされているチェックポイントが使用されます。

## インストール

```bash
pip install -U google-genai pydantic pillow httpx
```

リポジトリを取得した場合の例：

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
pip install -U google-genai pydantic pillow httpx
```

## Stable Diffusion WebUIの起動

APIを有効にして起動してください。

### Windows

```bat
webui-user.bat --api
```

### Linux / macOS

```bash
./webui.sh --api
```

既定のAPI URL：

```text
http://127.0.0.1:7860/sdapi/v1/txt2img
```

## 目標の指定

スクリプトと同じ作業ディレクトリへ、`GOAL.jpg` または `GOAL.txt` を置きます。

優先順位は次のとおりです。

```text
1. GOAL.jpg
2. GOAL.txt
3. どちらもなければエラー終了
```

両方が存在する場合は `GOAL.jpg` が優先されます。

### 参照画像を使う

```text
GOAL.jpg
```

Geminiが画像を分析し、次の形式の日本語目標へ変換します。

```text
SUBJECT:
...

REQUIRED:
...

STYLE:
...

MOOD:
...

AVOID:
...
```

### テキスト目標を使う

`GOAL.txt` にはPythonコードや引用符を入れず、目標本文だけを書きます。

```text
SUBJECT:
廃墟に立つ女性の天使戦士。

REQUIRED:
全身を覆う簡素な実用鎧を着用している。
巨大な剣を持っている。
大きな翼がある。
背景に廃墟、壊れた柱、瓦礫が認識できる。

STYLE:
人物と背景を同じ描画密度で扱った、未完成のデジタルラフスケッチ。
線は細く、部分的に輪郭が開いている。
清書や完成イラストには見えない。

MOOD:
荘厳な雰囲気。

AVOID:
下着姿、裸、ビキニアーマー、過度な露出。
人物だけが高密度に描き込まれた状態。
完成された線画、精密な建築描写、過度な装飾。
```

## 実行

環境変数にGemini APIキーを設定します。

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python main.py
```

### Windows コマンドプロンプト

```bat
set GEMINI_API_KEY=YOUR_API_KEY
python main.py
```

### Linux / macOS

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
python main.py
```

## 環境変数

### Gemini

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `GEMINI_API_KEY` | なし | Gemini APIキー |
| `GEMINI_PLANNER_MODEL` | `gemini-3-flash-preview` | Planner用モデル |
| `GEMINI_EVALUATOR_MODEL` | `gemini-3-flash-preview` | Evaluator用モデル |
| `GEMINI_GOAL_MODEL` | Plannerと同じ | 画像分析・GoalCompiler用モデル |

モデル名は利用可能なGeminiモデルに合わせて変更してください。

### 入力ファイル

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `GOAL_IMAGE_PATH` | `GOAL.jpg` | 参照画像のパス |
| `GOAL_TEXT_PATH` | `GOAL.txt` | テキスト目標のパス |

### Stable Diffusion

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `SD_API_URL` | `http://127.0.0.1:7860/sdapi/v1/txt2img` | txt2img API URL |
| `SD_STEPS` | `20` | Sampling steps |
| `SD_WIDTH` | `768` | 画像幅 |
| `SD_HEIGHT` | `1024` | 画像高さ |
| `SD_CFG` | `4.0` | CFG scale |
| `SD_SAMPLER` | `Euler a` | Sampler |
| `SD_SCHEDULER` | 未指定 | Scheduler |
| `SD_SEED` | `-1` | 初回Seed。`-1`は初回ランダム、その後固定 |

### ループ

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `MIN_ITERATIONS` | `3` | 合格しても最低限実行する回数 |
| `MAX_ITERATIONS` | `10` | 最大反復回数 |
| `COOLDOWN_SECONDS` | `3` | 反復間の待機秒数 |
| `INFRA_RETRY_COUNT` | `2` | 通信障害時の追加再試行回数 |
| `INFRA_RETRY_DELAY_SECONDS` | `5` | 通信再試行までの待機秒数 |

### 評価

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `GOAL_COMPLIANCE_WEIGHT` | `0.65` | テキスト目標適合スコアの重み |
| `REFERENCE_STRUCTURE_WEIGHT` | `0.35` | 参照画像構造スコアの重み |
| `MIN_REFERENCE_STRUCTURE_SCORE` | `0.80` | 参照画像使用時の最低構造スコア |
| `MIN_OVERALL_SCORE` | `0.85` | 合格に必要な最低総合スコア |
| `MIN_CRITERION_SCORE` | `0.82` | 重要criterionに必要な最低スコア |

参照画像を使用しない場合、参照画像構造スコアは総合評価へ含まれません。

## 参照画像比較のルール

参照画像比較は、元画像をそのまま模写するためのピクセル比較ではありません。次の構造的特徴だけを比較します。

- キャンバス内の構図
- 主体の位置と相対サイズ
- ポーズと向き
- 大きなシルエット
- 武器や主要物体の位置、大きさ、角度
- 背景の大きな形状と配置
- 余白と空間関係

次の要素は参照画像比較から除外され、`TARGET_GOAL` のSTYLE条件だけで評価されます。

- 色、配色、線色
- 線の粗さや滑らかさ
- 塗り方やレンダリング技法
- 陰影
- 描画密度
- 細部の装飾
- 完成度
- 写実、アニメ、ラフ、線画などの画風

そのため、参照画像が完成カラーイラストでも、生成目標をモノクロのラフ線画へ変更できます。

## Seedの動作

`SD_SEED=-1` の場合：

1. 1回目だけランダムSeedで生成
2. Stable Diffusion APIの応答から実際のSeedを取得
3. 2回目以降は同じSeedを固定

これにより、Prompt変更による差とSeedのばらつきを分離しやすくなります。

最初から固定したい場合：

```bash
SD_SEED=123456789
```

## 人間による評価操作

各反復のAI評価後に、次の入力を選択できます。

```text
Enter = AI評価を採用
f     = 次回への改善指示を追加
q     = 人間判断で合格
```

最低反復回数より前に `q` を選択しても、ループエンジンは最低反復回数まで実行します。

## 出力

実行ごとに開始日時のディレクトリが作成されます。

```text
outputs/
└── 20260724_120000/
    ├── generated_target_goal.txt
    ├── effective_target_goal.txt
    ├── compiled_goal.json
    ├── initial_prompts.txt
    ├── prompt_iter_01.txt
    ├── generation_info_iter_01.json
    ├── evaluate_iter_01.png
    ├── output_iter_01.png
    └── ...
```

- `generated_target_goal.txt`: `GOAL.jpg` から生成した目標。画像使用時のみ
- `effective_target_goal.txt`: GoalCompilerへ実際に渡した目標
- `compiled_goal.json`: criteriaと初期Prompt
- `initial_prompts.txt`: GoalCompilerが作成した初期Positive / Negative Prompt
- `prompt_iter_XX.txt`: 各反復で実際に使用したPromptとSeed
- `generation_info_iter_XX.json`: Stable Diffusion APIの生成情報
- `evaluate_iter_XX.png`: 評価対象として保存した画像
- `output_iter_XX.png`: 各反復の生成結果

## 推奨ディレクトリ構成

```text
.
├── main.py
├── GOAL.jpg                 # 任意。ある場合はこちらを優先
├── GOAL.txt                 # GOAL.jpgがない場合に使用
├── README.md
└── outputs/                 # 実行時に生成
```

## 注意事項

- Stable Diffusionの使用モデルはAPI payloadで固定されていません。WebUIで正しいモデルをロードしてください。
- API利用料金、ローカルGPU負荷、生成時間に注意してください。

