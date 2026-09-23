"use client";

export type QueryMeta = {
    retrieval_ms: number;
    results_count: number;
    cache_hit: boolean;
    reranked: boolean;
};

export function TimingBar({ meta }: { meta: QueryMeta }) {
    const seconds = (meta.retrieval_ms / 1000).toFixed(1);

    return (
        <div style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 6,
            marginBottom: 12,
            animation: "fadeIn 0.3s ease",
        }}>
            <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .timing-pill {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 3px 8px;
          border-radius: 20px;
          font-size: 10px;
          font-weight: 600;
          font-family: 'SF Mono', 'Fira Code', monospace;
          border: 1px solid;
          white-space: nowrap;
        }
      `}</style>
            {meta.retrieval_ms > 0 && (
                <span className="timing-pill" style={{
                    background: "rgba(232,96,38,0.08)",
                    borderColor: "rgba(232,96,38,0.3)",
                    color: "#e86026",
                }}>
                    ⚡ {seconds}s retrieval(Free-Tier)
                </span>)}

            <span className="timing-pill" style={{
                background: "var(--bg-3)",
                borderColor: "var(--border)",
                color: "var(--text-3)",
            }}>
                {meta.results_count} spots
            </span>

            {meta.reranked && (
                <span className="timing-pill" style={{
                    background: "var(--bg-3)",
                    borderColor: "var(--border)",
                    color: "var(--text-3)",
                }}>
                    ↑ reranked
                </span>
            )}

            {/* <span className="timing-pill" style={{
                background: meta.cache_hit ? "rgba(34,197,94,0.08)" : "var(--bg-3)",
                borderColor: meta.cache_hit ? "rgba(34,197,94,0.3)" : "var(--border)",
                color: meta.cache_hit ? "#22c55e" : "var(--text-3)",
            }}>
                {meta.cache_hit ? "✓ cached" : "cache miss"}
            </span> */}
        </div>
    );
}