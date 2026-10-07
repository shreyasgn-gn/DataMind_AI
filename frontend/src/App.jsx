import { useEffect, useMemo, useState } from "react";
import "./App.css";

const ANALYSIS_STAGES = [
  "Inspecting dataset",
  "Understanding your question",
  "Checking data quality",
  "Choosing the analysis strategy",
  "Engineering features",
  "Running the analysis",
  "Generating explanations",
];

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return String(value);
  }

  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function StatusDot({ online }) {
  return (
    <span className={`status-dot ${online ? "online" : "offline"}`} />
  );
}

function MetricCard({ label, value, detail, accent = "default" }) {
  return (
    <div className={`metric-card ${accent}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  );
}

function FeatureBar({ feature, importance, maxImportance }) {
  const percentage =
    maxImportance > 0
      ? Math.max(2, (Number(importance) / maxImportance) * 100)
      : 2;

  return (
    <div className="feature-item">
      <div className="feature-meta">
        <span>{feature}</span>
        <strong>{formatNumber(importance, 4)}</strong>
      </div>

      <div className="feature-track">
        <div
          className="feature-fill"
          style={{ width: `${Math.min(100, percentage)}%` }}
        />
      </div>
    </div>
  );
}

function ClusterBar({ cluster, count, maxCount }) {
  const width =
    maxCount > 0
      ? Math.max(4, (Number(count) / maxCount) * 100)
      : 4;

  return (
    <div className="cluster-item">
      <div className="cluster-meta">
        <span>Cluster {cluster}</span>
        <strong>{formatNumber(count, 0)} rows</strong>
      </div>

      <div className="cluster-track">
        <div
          className="cluster-fill"
          style={{ width: `${Math.min(100, width)}%` }}
        />
      </div>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      try {
        const response = await fetch("/api/health", {
          method: "GET",
          cache: "no-store",
        });

        if (!cancelled) {
          setBackendOnline(response.ok);
        }
      } catch {
        if (!cancelled) {
          setBackendOnline(false);
        }
      }
    };

    checkBackend();

    const interval = setInterval(checkBackend, 10000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!loading) {
      setLoadingStage(0);
      return undefined;
    }

    const interval = setInterval(() => {
      setLoadingStage((current) =>
        Math.min(
          current + 1,
          ANALYSIS_STAGES.length - 1
        )
      );
    }, 2200);

    return () => clearInterval(interval);
  }, [loading]);

  const taskType =
    result?.task?.task_type ||
    result?.task?.problem_type ||
    "";

  const isSupervised =
    taskType === "classification" ||
    taskType === "regression";

  const isClustering =
    taskType === "clustering";

  const isAnomalyDetection =
    taskType === "anomaly_detection";

  const topFeatures = useMemo(
    () => result?.explainability?.top_features || [],
    [result]
  );

  const shapFeatures = useMemo(
    () =>
      result?.explainability?.shap_top_features || [],
    [result]
  );

  const clusterSizes = useMemo(() => {
    const sizes =
      result?.unsupervised?.cluster_sizes || {};

    return Object.entries(sizes).sort(
      ([a], [b]) => Number(a) - Number(b)
    );
  }, [result]);

  const topAnomalies = useMemo(
    () =>
      result?.unsupervised?.top_anomalies || [],
    [result]
  );

  const maxFeatureImportance = useMemo(() => {
    return Math.max(
      ...topFeatures.map(
        (item) =>
          Number(item.importance) || 0
      ),
      0
    );
  }, [topFeatures]);

  const maxShapImportance = useMemo(() => {
    return Math.max(
      ...shapFeatures.map(
        (item) =>
          Number(item.importance) || 0
      ),
      0
    );
  }, [shapFeatures]);

  const maxClusterCount = useMemo(() => {
    return Math.max(
      ...clusterSizes.map(
        ([, count]) => Number(count) || 0
      ),
      0
    );
  }, [clusterSizes]);

  const analyzeDataset = async () => {
    if (!file) {
      setError("Please select a CSV or XLSX file.");
      return;
    }

    if (!question.trim()) {
      setError(
        "Please enter a question for DataMind AI."
      );
      return;
    }

    setLoading(true);
    setLoadingStage(0);
    setError("");
    setResult(null);
    setCopied(false);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      formData.append(
        "question",
        question.trim()
      );

      const response = await fetch(
        "/api/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      const rawText =
        await response.text();

      let data = null;

      if (rawText.trim()) {
        try {
          data = JSON.parse(
            rawText
          );
        } catch {
          throw new Error(
            `The server returned an invalid response (${response.status}).`
          );
        }
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            `Analysis failed with status ${response.status}.`
        );
      }

      if (!data) {
        throw new Error(
          "The server returned an empty response."
        );
      }

      if (data.status === "failed") {
        throw new Error(
          data.errors?.[0] ||
            "DataMind could not complete the analysis."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to complete the analysis."
      );
    } finally {
      setLoading(false);
    }
  };

  const copyReport = async () => {
    if (!result?.final_answer) {
      return;
    }

    try {
      await navigator.clipboard.writeText(
        result.final_answer
      );

      setCopied(true);

      setTimeout(
        () => setCopied(false),
        1800
      );
    } catch {
      setError(
        "Could not copy the report."
      );
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setError("");
    setQuestion("");
    setFile(null);
    setCopied(false);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            D
          </div>

          <div>
            <h1>DataMind AI</h1>
            <p>
              Autonomous AI Data Scientist
            </p>
          </div>
        </div>

        <div className="topbar-right">
          <div className="engine-chip">
            <StatusDot
              online={backendOnline}
            />

            <span>
              {backendOnline
                ? "AI Engine Online"
                : "AI Engine Offline"}
            </span>
          </div>

          <div className="local-chip">
            LOCAL • OLLAMA
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">
            AUTONOMOUS DATA SCIENCE
          </div>

          <h2>
            From raw data to
            <span>
              intelligent decisions.
            </span>
          </h2>

          <p>
            Upload a dataset. Ask a question
            in plain English. DataMind AI
            chooses the right analysis,
            executes the workflow, and
            explains what it finds.
          </p>

          <div className="hero-stats">
            <span>CSV + XLSX</span>
            <span>LOCAL LLM</span>
            <span>ML + SHAP</span>
            <span>CLUSTERING</span>
            <span>ANOMALY DETECTION</span>
          </div>
        </section>

        <section className="workspace">
          <div className="control-card">
            <div className="card-heading">
              <div>
                <div className="section-kicker">
                  WORKSPACE
                </div>

                <h3>
                  Start an analysis
                </h3>

                <p>
                  Give DataMind a dataset
                  and a goal.
                </p>
              </div>

              {result && (
                <button
                  className="ghost-button"
                  onClick={resetAnalysis}
                >
                  New analysis
                </button>
              )}
            </div>

            <label className="upload-box">
              <input
                type="file"
                accept=".csv,.xlsx"
                onChange={(event) => {
                  setFile(
                    event.target.files?.[0] ||
                      null
                  );

                  setError("");
                }}
              />

              <div className="upload-icon">
                <span>↑</span>
              </div>

              {file ? (
                <>
                  <strong>
                    {file.name}
                  </strong>

                  <span>
                    Dataset ready for analysis
                  </span>
                </>
              ) : (
                <>
                  <strong>
                    Upload your dataset
                  </strong>

                  <span>
                    CSV or XLSX • click to browse
                  </span>
                </>
              )}
            </label>

            <div className="field-label-row">
              <label
                className="field-label"
                htmlFor="question"
              >
                Ask DataMind AI
              </label>

              <span className="field-hint">
                Natural language
              </span>
            </div>

            <textarea
              id="question"
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target.value
                )
              }
              placeholder="Example: predicted profit"
              rows="4"
            />

            <div className="prompt-list">
              <button
                type="button"
                onClick={() =>
                  setQuestion(
                    "predicted profit"
                  )
                }
              >
                Predict a target
              </button>

              <button
                type="button"
                onClick={() =>
                  setQuestion(
                    "find important factors affecting the target"
                  )
                }
              >
                Find important factors
              </button>

              <button
                type="button"
                onClick={() =>
                  setQuestion(
                    "group customers into similar segments"
                  )
                }
              >
                Find similar groups
              </button>

              <button
                type="button"
                onClick={() =>
                  setQuestion(
                    "find unusual or anomalous records"
                  )
                }
              >
                Find anomalies
              </button>
            </div>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

            <button
              className="analyze-button"
              onClick={analyzeDataset}
              disabled={loading}
            >
              <span>
                {loading
                  ? "Analyzing dataset"
                  : "Run Analysis"}
              </span>

              <span>
                {loading
                  ? "…"
                  : "→"}
              </span>
            </button>

            <div className="privacy-note">
              <span>◉</span>

              <span>
                Your dataset is processed
                inside your local DataMind
                environment.
              </span>
            </div>
          </div>

          <div className="results-card">
            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-orbit">
                  <div className="empty-core">
                    ✦
                  </div>
                </div>

                <div className="section-kicker">
                  ANALYSIS CONSOLE
                </div>

                <h3>
                  Insights will appear here
                </h3>

                <p>
                  DataMind automatically
                  chooses between exploration,
                  prediction, clustering, and
                  anomaly detection based on
                  your request.
                </p>

                <div className="empty-pipeline">
                  <span>Inspect</span>
                  <i />
                  <span>Decide</span>
                  <i />
                  <span>Analyze</span>
                  <i />
                  <span>Explain</span>
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="loading-ring">
                  <span />
                </div>

                <div className="section-kicker">
                  AI ANALYSIS RUNNING
                </div>

                <h3>
                  {ANALYSIS_STAGES[
                    loadingStage
                  ]}
                </h3>

                <p>
                  DataMind is executing the
                  autonomous data-science
                  pipeline.
                </p>

                <div className="stage-list">
                  {ANALYSIS_STAGES.map(
                    (
                      stage,
                      index
                    ) => (
                      <div
                        className={`stage-row ${
                          index <
                          loadingStage
                            ? "complete"
                            : index ===
                                loadingStage
                              ? "active"
                              : ""
                        }`}
                        key={stage}
                      >
                        <span className="stage-indicator">
                          {index <
                          loadingStage
                            ? "✓"
                            : index + 1}
                        </span>

                        <span>
                          {stage}
                        </span>

                        {index ===
                          loadingStage && (
                          <b>
                            Running
                          </b>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div className="result-top">
                  <div>
                    <div className="result-kicker">
                      ANALYSIS COMPLETE
                    </div>

                    <h3>
                      {result.file_name}
                    </h3>

                    <div className="result-meta">
                      <span>
                        {formatNumber(
                          result.dataset
                            ?.rows,
                          0
                        )}{" "}
                        rows
                      </span>

                      <span>•</span>

                      <span>
                        {formatNumber(
                          result.dataset
                            ?.columns,
                          0
                        )}{" "}
                        columns
                      </span>

                      <span>•</span>

                      <span>
                        {taskType.replace(
                          "_",
                          " "
                        )}
                      </span>
                    </div>
                  </div>

                  <div className="result-actions">
                    <span className="complete-chip">
                      <StatusDot
                        online
                      />
                      Completed
                    </span>

                    <button
                      className="ghost-button"
                      onClick={copyReport}
                    >
                      {copied
                        ? "Copied"
                        : "Copy report"}
                    </button>
                  </div>
                </div>

                <div className="summary-strip">
                  <div>
                    <span>
                      ANALYSIS TYPE
                    </span>

                    <strong>
                      {taskType.replace(
                        "_",
                        " "
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      TARGET
                    </span>

                    <strong>
                      {result.task
                        ?.target_column ||
                        "Not required"}
                    </strong>
                  </div>

                  <div>
                    <span>
                      ENGINE
                    </span>

                    <strong>
                      {result.task
                        ?.model ||
                        result.unsupervised
                          ?.method ||
                        "DataMind"}
                    </strong>
                  </div>

                  <div>
                    <span>
                      SOURCE
                    </span>

                    <strong>
                      {result.task
                        ?.source ||
                        "—"}
                    </strong>
                  </div>
                </div>

                {isSupervised && (
                  <>
                    <div className="metric-grid">
                      <MetricCard
                        label="R² SCORE"
                        value={formatNumber(
                          result.evaluation
                            ?.r2,
                          4
                        )}
                        detail="Explained variance"
                        accent="primary"
                      />

                      <MetricCard
                        label="MAE"
                        value={formatNumber(
                          result.evaluation
                            ?.mae,
                          2
                        )}
                        detail="Mean absolute error"
                      />

                      <MetricCard
                        label="RMSE"
                        value={formatNumber(
                          result.evaluation
                            ?.rmse,
                          2
                        )}
                        detail="Root mean squared error"
                      />

                      <MetricCard
                        label="SHAP"
                        value={
                          result.explainability
                            ?.shap_available
                            ? "READY"
                            : "N/A"
                        }
                        detail="Model explainability"
                        accent={
                          result.explainability
                            ?.shap_available
                            ? "success"
                            : "default"
                        }
                      />
                    </div>

                    <div className="insight-grid">
                      <div className="insight-card">
                        <div className="insight-card-header">
                          <div>
                            <div className="section-kicker">
                              DATA HEALTH
                            </div>

                            <h4>
                              Quality snapshot
                            </h4>
                          </div>

                          <span
                            className={
                              result.data_quality
                                ?.has_quality_issues
                                ? "warning-badge"
                                : "success-badge"
                            }
                          >
                            {result
                              .data_quality
                              ?.has_quality_issues
                              ? "Needs attention"
                              : "Healthy"}
                          </span>
                        </div>

                        <div className="snapshot-grid">
                          <div>
                            <strong>
                              {formatNumber(
                                result
                                  .data_quality
                                  ?.missing_values,
                                0
                              )}
                            </strong>

                            <span>
                              Missing values
                            </span>
                          </div>

                          <div>
                            <strong>
                              {formatNumber(
                                result
                                  .data_quality
                                  ?.duplicate_rows,
                                0
                              )}
                            </strong>

                            <span>
                              Duplicate rows
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="insight-card">
                        <div className="insight-card-header">
                          <div>
                            <div className="section-kicker">
                              LEAKAGE GUARD
                            </div>

                            <h4>
                              Training protection
                            </h4>
                          </div>

                          <span
                            className={
                              result.leakage
                                ?.detected
                                ? "warning-badge"
                                : "success-badge"
                            }
                          >
                            {result.leakage
                              ?.detected
                              ? "Handled"
                              : "Clear"}
                          </span>
                        </div>

                        <p className="insight-text">
                          {result.leakage
                            ?.detected
                            ? "DataMind detected potentially derived information and prevented it from contaminating model training."
                            : "No confirmed target leakage was detected."}
                        </p>

                        {result.leakage
                          ?.removed_columns
                          ?.length >
                          0 && (
                          <div className="tag-row">
                            {result.leakage.removed_columns.map(
                              (
                                column
                              ) => (
                                <span
                                  className="data-tag danger"
                                  key={
                                    column
                                  }
                                >
                                  Removed:{" "}
                                  {
                                    column
                                  }
                                </span>
                              )
                            )}

                            {result.leakage
                              .excluded_columns
                              ?.map(
                                (
                                  column
                                ) => (
                                  <span
                                    className="data-tag warning"
                                    key={
                                      column
                                    }
                                  >
                                    Excluded:{" "}
                                    {
                                      column
                                    }
                                  </span>
                                )
                              )}
                          </div>
                        )}
                      </div>
                    </div>

                    {topFeatures.length >
                      0 && (
                      <div className="analysis-panel">
                        <div className="panel-heading">
                          <div>
                            <div className="section-kicker">
                              MODEL EXPLAINABILITY
                            </div>

                            <h4>
                              What drives the prediction?
                            </h4>
                          </div>

                          <span className="panel-pill">
                            {
                              topFeatures.length
                            }{" "}
                            features
                          </span>
                        </div>

                        <div className="feature-list">
                          {topFeatures
                            .slice(
                              0,
                              8
                            )
                            .map(
                              (
                                item
                              ) => (
                                <FeatureBar
                                  key={
                                    item.feature
                                  }
                                  feature={
                                    item.feature
                                  }
                                  importance={
                                    item.importance
                                  }
                                  maxImportance={
                                    maxFeatureImportance
                                  }
                                />
                              )
                            )}
                        </div>
                      </div>
                    )}

                    {shapFeatures.length >
                      0 && (
                      <div className="analysis-panel">
                        <div className="panel-heading">
                          <div>
                            <div className="section-kicker">
                              SHAP EXPLANATION
                            </div>

                            <h4>
                              Global contribution strength
                            </h4>
                          </div>

                          <span className="panel-pill">
                            SHAP enabled
                          </span>
                        </div>

                        <div className="feature-list">
                          {shapFeatures
                            .slice(
                              0,
                              6
                            )
                            .map(
                              (
                                item
                              ) => (
                                <FeatureBar
                                  key={
                                    item.feature
                                  }
                                  feature={
                                    item.feature
                                  }
                                  importance={
                                    item.importance
                                  }
                                  maxImportance={
                                    maxShapImportance
                                  }
                                />
                              )
                            )}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {isClustering && (
                  <>
                    <div className="metric-grid">
                      <MetricCard
                        label="CLUSTERS"
                        value={formatNumber(
                          result.unsupervised
                            ?.cluster_count,
                          0
                        )}
                        detail="Groups identified"
                        accent="primary"
                      />

                      <MetricCard
                        label="SILHOUETTE"
                        value={formatNumber(
                          result.unsupervised
                            ?.silhouette_score,
                          4
                        )}
                        detail="Cluster separation"
                      />

                      <MetricCard
                        label="QUALITY"
                        value={
                          result.unsupervised
                            ?.cluster_quality ||
                          "—"
                        }
                        detail="Separation assessment"
                      />

                      <MetricCard
                        label="ROWS ANALYZED"
                        value={formatNumber(
                          result.unsupervised
                            ?.rows_analyzed,
                          0
                        )}
                        detail="Usable observations"
                      />
                    </div>

                    <div className="analysis-panel">
                      <div className="panel-heading">
                        <div>
                          <div className="section-kicker">
                            CLUSTERING
                          </div>

                          <h4>
                            Cluster distribution
                          </h4>
                        </div>

                        <span className="panel-pill">
                          KMeans
                        </span>
                      </div>

                      <div className="cluster-list">
                        {clusterSizes.map(
                          ([
                            cluster,
                            count,
                          ]) => (
                            <ClusterBar
                              key={
                                cluster
                              }
                              cluster={
                                cluster
                              }
                              count={
                                count
                              }
                              maxCount={
                                maxClusterCount
                              }
                            />
                          )
                        )}
                      </div>
                    </div>

                    {result.unsupervised
                      ?.warning && (
                      <div className="notice-card warning-notice">
                        <strong>
                          Interpretation note
                        </strong>

                        <p>
                          {
                            result
                              .unsupervised
                              .warning
                          }
                        </p>
                      </div>
                    )}
                  </>
                )}

                {isAnomalyDetection && (
                  <>
                    <div className="metric-grid">
                      <MetricCard
                        label="ANOMALIES"
                        value={formatNumber(
                          result.unsupervised
                            ?.anomaly_count,
                          0
                        )}
                        detail="Potential unusual rows"
                        accent="primary"
                      />

                      <MetricCard
                        label="ANOMALY RATE"
                        value={`${(
                          Number(
                            result.unsupervised
                              ?.anomaly_rate ||
                              0
                          ) * 100
                        ).toFixed(2)}%`}
                        detail="Of analyzed rows"
                      />

                      <MetricCard
                        label="NORMAL"
                        value={formatNumber(
                          result.unsupervised
                            ?.normal_count,
                          0
                        )}
                        detail="Rows within threshold"
                      />

                      <MetricCard
                        label="METHOD"
                        value="IF"
                        detail="Isolation Forest"
                      />
                    </div>

                    <div className="analysis-panel">
                      <div className="panel-heading">
                        <div>
                          <div className="section-kicker">
                            ANOMALY DETECTION
                          </div>

                          <h4>
                            Highest anomaly scores
                          </h4>
                        </div>

                        <span className="panel-pill">
                          Top 10
                        </span>
                      </div>

                      <div className="anomaly-table">
                        <div className="anomaly-header">
                          <span>
                            Row index
                          </span>

                          <span>
                            Score
                          </span>

                          <span>
                            Status
                          </span>
                        </div>

                        {topAnomalies.map(
                          (item) => (
                            <div
                              className="anomaly-row"
                              key={
                                item.row_index
                              }
                            >
                              <span>
                                #
                                {
                                  item.row_index
                                }
                              </span>

                              <strong>
                                {Number(
                                  item.anomaly_score
                                ).toFixed(
                                  6
                                )}
                              </strong>

                              <span className="anomaly-badge">
                                Potential anomaly
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </>
                )}

                <div className="report-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="section-kicker">
                        AI REPORT
                      </div>

                      <h4>
                        DataMind findings
                      </h4>
                    </div>

                    <span className="panel-pill">
                      {result.final_answer_source ||
                        "DataMind"}
                    </span>
                  </div>

                  <pre>
                    {result.final_answer ||
                      "No report generated."}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;