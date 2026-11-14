# colmat-rs に対応する Julia 実験室

Rust の `eigh`・`qr`・`svd` のどの段階に改善の余地があるかを、
Float64 の最小実装で調べるためのコードです。Rust の対応基準は
`b292f4746d5ae9964649bd7224ee7824a831a145`。

| 対象 | 自前の Julia 実装 | Rust との違い・既存ソルバとの境界 |
| --- | --- | --- |
| QR | Householder パネル、compact WY によるブロック後続更新、ブロック形式での薄い Q の生成 | Rust の再帰的パネル・look-ahead は未実装。ブロック幅を指定可能 |
| eigh | 対称 Householder 三重対角化、ブロック形式の Q の生成、固有ベクトルの逆変換 | Rust の下三角融合カーネルと異なり、全三角を保持して matvec と rank-2 更新を別々に実行。最終解法は標準 `eigen(SymTridiagonal)` |
| SVD | 遅延補正 `A − VYᵀ − XUᵀ` によるブロック二重対角化、ブロック形式での左右の基底・特異ベクトルの生成 | Rust 同様に幅16・残り128以下で非ブロック処理。1 KiB 倍数の列ストライドは8行パディング。最終解法は標準 `svd(Bidiagonal)` |
| eigh の修復 | 重み再計算と固有ベクトル列の構築 | Rust の今回の改善部分を式・スカラー演算順序を保って移植。secular equation の根は fixture 作成時に既存ソルバで計算し、測定から除外 |

`qr_lab`・`svd_lab` は薄いベクトルを返します。SVD の values-only/dqds、
複素数、Rust の SIMD 命令・コストモデル・スクラッチ量の再現は対象外です。
Julia 1.13.3 の標準三重対角ソルバは `stegr`、二重対角 SVD は `bdsdc` を呼びます。
Rust の固有値・特異値ソルバそのものの性能限界を、これらから直接結論づけることはできません。

## 実行

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
taskset -c 1-16 julia --project=julia -t 16,0 julia/test/runtests.jl
```

登録済み ThreadTeams.jl 1.4.0 を使用し、依存バージョンは Manifest.toml に固定しています。
確認した環境は Julia 1.13.3。Project の下限1.10での実行は未検証です。

```julia
include("julia/src/ColmatLab.jl")
using .ColmatLab, LinearAlgebra
BLAS.set_num_threads(1)             # 呼び出し側で一度設定
A = randn(256, 256)
ex = Team(8)                      # julia -t 8,0 以上で起動
F = eigh_lab(Matrix(Symmetric(A)), ex)
G = qr_lab(A, ex; block=16)
S = svd_lab(A, ex; block=16, floor=128, pad=true)
```

`Serial()` は同じ自前実装を直列で実行します。`Team(p; minwork=0)` は
ThreadTeams の常駐ワーカーに出力範囲を静的に分割します。`minwork` は
並列化に必要な推定仕事量で、初期値0は閾値をチューニングしていない対照条件です。
これは Rust の自動並列幅選択モデルではありません。
BLAS は1スレッドにして、Julia ワーカーとの二重並列化を避けてください。

`QRWorkspace`・`TridiagWorkspace`・`BidiagWorkspace` を確保して、対応する
`qr_factor!`・`tridiagonalize!`・`bidiagonalize!` に再利用できます。
これらの低水準カーネルは有限で適度にスケールされた入力、同じ寸法の作業領域、
競合しない呼び出しを前提とします。行列を破壊し、変換ベクトルを packed 形式で保存します。
高水準ドライバは入力をコピー・正規化し、出力と作業領域を毎回確保します。
Rust の per-call allocation-free 契約を再現するものではありません。

## CPU を固定した参考計測

```sh
# 出力先は新しいディレクトリを指定する。既存結果は上書きしない。
WIDTHS=1,4,8,16 SIZES=256,512 REPS=7 \
  taskset -c 1-16 julia --project=julia -t 16,0 \
  julia/measure.jl /tmp/colmat-julia-reference
python3 julia/summarize.py /tmp/colmat-julia-reference

# ホスト計測ロックと前後のCPU負荷観測も記録するLinux用ランチャー
WIDTHS=1,4,8,16 SIZES=256,512 REPS=7 \
  python3 julia/run_pinned.py /tmp/colmat-julia-pinned
```

このマシン `witch` の CPU ID 1〜16 は16個の別々の物理コアです。
SMT 兄弟は19〜34。CPU17/35はこの実験では使いません。
`taskset` は全ワーカーの実行可能範囲を1〜16に制限します。
各ワーカーを特定の1コアに個別固定する設定ではありません。
1スレッドの試行もこの範囲内で移動できるため、厳密な同一コア比較には
別プロセスを個別 CPU 集合で起動してください。htop は0始まりのIDを確認してください。

`samples.csv` は全サンプルの時間、割り当てバイト数、GC時間、段階別時間を保持します。
`metadata.txt` は Julia、BLAS、CPU、実行可能CPU、条件を記録します。
`run_pinned.py` は隣の colmat-benchmark の campaign.lock を保持し、
前後3秒の測定コアとSMT兄弟の負荷を記録します。別ホストでは `--lock` を指定できます。
負荷条件が悪くても結果を残し、再試行による選別はしません。
二回のウォームアップは測定から除外し、各サンプル直前に GC を実行します。
通常の ThreadTeams の待機設定を使い、`parallel_phase` による常時スピンは行いません。

- `*_kernel`: 作業領域を再利用。入力のリセットは測定外。SVD はパディングした列ストライドを使用。
- `julia`: 自前実装全体。コピー、正規化、作業領域、出力の確保を含む。
- `stdlib`: 標準の密行列 `eigen`・`qr`・`svd`。同じデータとベクトル出力、BLASスレッド数は比較する幅に設定。
- 段階 a/b/c: QR は factor/basis/0、eigh・SVD は reduction/terminal solver/backtransform。全体時間にはこれら以外の準備時間も含む。

単一プロセスの走査は探索用です。幅の順序、ターボ、他プロセス、共有L3、SMT兄弟の負荷を
分離した受け入れ試験ではありません。静かな条件の独立プロセスで順序を反転して繰り返し、
実行時間のばらつきを超える差を確認してください。Rust と同一入力・同一出力・同一予算の
paired 比較はまだ組み込んでいません。

## 改善余地の読み方

1. 同じ自前カーネルの1Tと複数Tを比べ、並列化が効くサイズと幅を探す。
2. 全体時間と段階別時間で、その改善が分解全体にどこまで効くかを確認する。
3. QR の `block`、SVD の `block`/`floor`/`pad`、Team の `minwork` を一つずつ変える。
4. Julia の方が遅くても Rust が限界に達した証拠にはならない。反対に速くても、
   別の最終ソルバや BLAS、演算量の違いを除外するまでは Rust の改善率とは呼ばない。

特に tridiagonalization の別パス更新、QR の look-ahead 不在、SVD の補正ループ、
共有基底を各 GEMM タスクが読む・パックするコストは区別が必要です。
並列分割は出力を所有し、O(p) の部分行列や直列結合を追加しませんが、
共有基底のキャッシュ/パッキング費用は p とともに増える場合があります。
64コアでのスケールは未検証です。

## 正しさと出典

テストは空・長方形・ランク落ち・ゼロ・密集/gradedスペクトル・極端な全体スケール、
ブロック境界、再構成、固有対残差、直交性、順序、直列/並列の修復結果のビット一致、
主要カーネルの型推論を検査します。

アルゴリズムの対応先は Rust の `qr/no_pivoting/factor.rs`、`evd/tridiag.rs`、
`evd/tridiag_evd/repair.rs`、`svd/bidiag.rs`。修復式以外は Julia として独立に実装しています。
ブロック二重対角化の設計は Rust と同じ、Dongarra, Hammarling, Sorensen (1989),
“Block reduction of matrices to condensed forms for eigenvalue computations”
に基づくものです。compact WY は `Q = I − VTVᵀ` を明示的に構築します。
独自コードのライセンスはリポジトリの MIT ライセンスに従います。
ThreadTeams の使い方は [公式README](https://github.com/vchuravy/ThreadTeams.jl) を参照しました。
商用ライブラリの内部実装は参照していません。

## 保存した探索結果

[作業記録](../docs/worklogs/2026-11-14-julia-decomposition-lab.md) に、
保存先と判断をまとめています。最終の参考走査は256/512、1/4/8/16T、各7回。
全280サンプルの再利用カーネル/修復処理で割り当て0バイトを観測しました。
修復 fixture の非デフレーション数は `k = n - n÷10` です。

CPU2などの同時負荷で計測前後の負荷条件が不成立だったため、実行時間は探索値です。
このJulia単独走査では Rust と直接比較していません。直接比較は下の章を参照してください。
最終ソルバの Rust 相当実装は未検証です。
このデータから Rust の高速化率や性能限界を確定しません。

## colmat-rs と同一入力で直接比較する

```sh
# Rust の比較用ライブラリ。colmat の本番コードや公開 C API は変更しない。
CARGO_TARGET_DIR=target cargo build --release --locked \
  --manifest-path julia/colmat-bridge/Cargo.toml
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python3 julia/compare_pinned.py /tmp/colmat-vs-julia \
  --starts 3 --reps 5 --widths 1,4,8,16 --sizes 256,512
python3 julia/compare_summary.py /tmp/colmat-vs-julia
```

比較対象は `colmat`、自作 `julia_lab`、標準密行列ソルバ `julia_stdlib` の3種類です。
colmat はこのチェックアウトの owning API (`self_adjoint_eigen` / `qr` + thin Q /
`thin_svd`) を private FFI 経由で呼びます。標準 Julia SVD は V を明示的な行列にします。
同じ Julia 行列を Rust に貸し出すため、入力はバイト単位で同一です。
入力の SHA256 と、各呼び出しの残差・直交性を保存します。
バックエンドごとに別プロセスを使い、使用しない Rust/ThreadTeams のワーカーは起動しません。
起動時の OpenBLAS スレッド数は1とし、標準実装の試行だけ指定幅に設定します。

すべて固有ベクトル／薄い Q と R／薄い U・S・V を求め、作業領域と因子の確保を含めます。
Rust 結果の検査用コピーと解放、Julia の検査、事前GC、ワーカー初期化は測定外です。
入力ファイルの読み込み・乱数生成・コンパイルも測定外です。
Julia `@timed` は Rust のヒープ確保を数えられないため、その欄は `missing` とします。

各予算・バックエンドに新しい Julia プロセスを使い、CPU ID は **1〜予算数** に固定します。
1T=CPU1、4T=CPU1〜4、8T=CPU1〜8、16T=CPU1〜16です。
Rust と Julia のチームはプロセス内で再利用します。各試行の前には測定外の20ms待機を置きます。
同一プロセスでの実装切り替えは、OpenBLAS の待機ワーカーによる順序依存を観測したため
比較に採用しません。別プロセスの実装順序を回転し、スレッド数の走査順も開始回で反転します。
各プロセスは計測前に対象実装で入力を検証し、計測した呼び出しも毎回検証します。

集計は各バックエンドのプロセス内中央値を求め、同じ開始回・入力・幅に対応する
`Julia中央値 / colmat中央値` を作り、3開始回の比の中央値を取ります。
1より大きければcolmatが速く、小さければそのJulia実装が速い値です。
負荷確認が不成立のプロセスも削除せず、件数と各プロセスの範囲を残します。

これは実装全体の比較です。アルゴリズム、BLAS、Rust の既定チューニング、
Julia の規格化処理と GC なども差に含まれ、言語・コンパイラだけの比較ではありません。

[直接比較の結果と制約](../docs/worklogs/2026-11-14-colmat-julia-comparison.md) に、
分離後の全1,080サンプルと負荷によるばらつきを記録しています。
