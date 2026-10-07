import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const analyzeDataset = async () => {
    if (!file) {
      setError("Please select a CSV or XLSX file.");
      return;
    }

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", question);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || "Analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">D</div>
          <div>
            <h1>DataMind AI</h1>
            <span>Autonomous AI Data Scientist</span>
          </div>
        </div>

        <div className="status-pill">
          <span className="status-dot"></span>
          Local AI Ready
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="hero-badge">AI-POWERED DATA ANALYSIS</div>

          <h2>
            Turn your dataset into
            <span> intelligent insights.</span>
          </h2>

          <p>
            Upload your data, ask a natural-language question, and let
            DataMind AI inspect, clean, model, evaluate, and explain the
            results automatically.
          </p>
        </section>

        <section className="workspace">
          <div className="control-card">
            <div className="section-heading">
              <div>
                <h3>Start an analysis</h3>
                <p>Works with CSV and Excel datasets.</p>
              </div>
            </div>

            <label className="upload-box">
              <input
                type="file"
                accept=".csv,.xlsx"
                onChange={(e) => {
                  setFile(e.target.files[0] || null);
                  setError("");
                }}
              />

              <div className="upload-icon">↑</div>

              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <span>File selected successfully</span>
                </>
              ) : (
                <>
                  <strong>Drop your dataset here</strong>
                  <span>or click to browse CSV / XLSX</span>
                </>
              )}
            </label>

            <label className="field-label">Ask DataMind AI</label>

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Example: predicted profit"
              rows="4"
            />

            <div className="example-row">
              <button
                type="button"
                onClick={() => setQuestion("predicted profit")}
              >
                Predict a value
              </button>

              <button
                type="button"
                onClick={() =>
                  setQuestion("find important factors affecting the target")
                }
              >
                Find important factors
              </button>
            </div>

            {error && <div className="error-box">{error}</div>}

            <button
              className="analyze-button"
              onClick={analyzeDataset}
              disabled={loading}
            >
              {loading ? "DataMind is analyzing..." : "Run Analysis →"}
            </button>
          </div>

          <div className="results-card">
            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h3>Your analysis will appear here</h3>
                <p>
                  DataMind will show task detection, model performance,
                  leakage protection, explainability, and AI-generated
                  findings.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-state">
                <div className="loader"></div>
                <h3>Analyzing your dataset</h3>
                <p>
                  Inspecting data, detecting the task, engineering features,
                  training models, and generating explanations.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div className="result-header">
                  <div>
                    <span className="result-label">ANALYSIS COMPLETE</span>
                    <h3>{result.file_name}</h3>
                  </div>

                  <div className="completed-badge">Completed</div>
                </div>

                <div className="metric-grid">
                  <div className="metric">
                    <span>Task</span>
                    <strong>{result.task?.task_type || "—"}</strong>
                  </div>

                  <div className="metric">
                    <span>Target</span>
                    <strong>{result.task?.target_column || "—"}</strong>
                  </div>

                  <div className="metric">
                    <span>Best Model</span>
                    <strong>{result.model?.best_model || "—"}</strong>
                  </div>

                  <div className="metric">
                    <span>R²</span>
                    <strong>
                      {result.evaluation?.r2 !== undefined
                        ? result.evaluation.r2
                        : "—"}
                    </strong>
                  </div>
                </div>

                <div className="info-grid">
                  <div className="info-panel">
                    <h4>Dataset</h4>
                    <p>
                      {result.dataset?.rows || 0} rows ·{" "}
                      {result.dataset?.columns || 0} columns
                    </p>
                  </div>

                  <div className="info-panel">
                    <h4>Data Quality</h4>
                    <p>
                      {result.data_quality?.missing_values || 0} missing values
                      · {result.data_quality?.duplicate_rows || 0} duplicates
                    </p>
                  </div>

                  <div className="info-panel">
                    <h4>Leakage Protection</h4>
                    <p>
                      {result.leakage?.detected
                        ? "Leakage detected and handled"
                        : "No leakage detected"}
                    </p>
                  </div>

                  <div className="info-panel">
                    <h4>Explainability</h4>
                    <p>
                      {result.explainability?.shap_available
                        ? "SHAP available"
                        : "SHAP unavailable"}
                    </p>
                  </div>
                </div>

                <div className="report-panel">
                  <h4>AI Report</h4>
                  <pre>{result.final_answer}</pre>
                </div>

                {result.explainability?.top_features?.length > 0 && (
                  <div className="features-panel">
                    <h4>Top Features</h4>

                    {result.explainability.top_features
                      .slice(0, 5)
                      .map((item) => (
                        <div className="feature-row" key={item.feature}>
                          <span>{item.feature}</span>
                          <strong>
                            {Number(item.importance).toFixed(4)}
                          </strong>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;