"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import Link from "next/link";
import { ArrowLeft, Eye, Code, Github, RefreshCw, AlertCircle } from "lucide-react";


const README_URL =
    process.env.NEXT_PUBLIC_README_URL ||
    "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/README.md";

const IS_PLACEHOLDER = README_URL.includes("YOUR_USER");

export default function ReadmePage() {
    const [content, setContent] = useState("");
    const [loading, setLoading] = useState(!IS_PLACEHOLDER);
    const [error, setError] = useState<string | null>(null);
    const [view, setView] = useState<"rendered" | "raw">("rendered");

    useEffect(() => {
        if (IS_PLACEHOLDER) return;
        fetchReadme();
    }, []);

    function fetchReadme() {
        setLoading(true);
        setError(null);
        fetch(README_URL)
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status} — check your NEXT_PUBLIC_README_URL`);
                return r.text();
            })
            .then((text) => { setContent(text); setLoading(false); })
            .catch((e) => { setError(e.message); setLoading(false); });
    }

    return (
        <div style={{ flex: 1, overflowY: "auto", background: "var(--bg)" }}>
            <div style={{ maxWidth: 820, margin: "0 auto", padding: "28px 24px 60px" }}>

                {/* ── Toolbar ── */}
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 28, flexWrap: "wrap" }}>
                    <Link
                        href="/"
                        style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 13px", borderRadius: 9, background: "var(--bg-2)", border: "1px solid var(--border-2)", fontSize: 13, fontWeight: 500, color: "var(--text-2)", textDecoration: "none", transition: "all 0.15s" }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--text)"; }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--text-2)"; }}
                    >
                        <ArrowLeft style={{ width: 14, height: 14 }} />
                        Back to Chat
                    </Link>

                    {/* View toggle */}
                    <div style={{ display: "flex", alignItems: "center", gap: 3, padding: 4, borderRadius: 10, background: "var(--bg-3)", border: "1px solid var(--border)", marginLeft: "auto" }}>
                        {[
                            { id: "rendered", Icon: Eye, label: "Preview" },
                            { id: "raw", Icon: Code, label: "Raw" },
                        ].map(({ id, Icon, label }) => (
                            <button
                                key={id}
                                onClick={() => setView(id as "rendered" | "raw")}
                                style={{
                                    display: "flex", alignItems: "center", gap: 6,
                                    padding: "6px 12px", borderRadius: 7,
                                    fontSize: 12, fontWeight: 500, cursor: "pointer",
                                    transition: "all 0.15s",
                                    border: view === id ? "1px solid var(--border-2)" : "1px solid transparent",
                                    background: view === id ? "var(--bg-2)" : "transparent",
                                    color: view === id ? "var(--text)" : "var(--text-2)",
                                    boxShadow: view === id ? "var(--shadow)" : "none",
                                }}
                            >
                                <Icon style={{ width: 13, height: 13 }} />
                                {label}
                            </button>
                        ))}
                    </div>

                    {/* GitHub link */}
                    {!IS_PLACEHOLDER && (
                        <a
                            href={README_URL
                                .replace("raw.githubusercontent.com", "github.com")
                                .replace("/main/README.md", "")
                                .replace("/master/README.md", "")}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 13px", borderRadius: 9, background: "var(--bg-2)", border: "1px solid var(--border-2)", fontSize: 12, fontWeight: 500, color: "var(--text-2)", textDecoration: "none", transition: "all 0.15s" }}
                            onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--text)"; }}
                            onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--text-2)"; }}
                        >
                            <Github style={{ width: 14, height: 14 }} />
                            View on GitHub
                        </a>
                    )}

                    {/* Refresh */}
                    {!IS_PLACEHOLDER && !loading && (
                        <button
                            onClick={fetchReadme}
                            style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 13px", borderRadius: 9, background: "var(--bg-2)", border: "1px solid var(--border-2)", fontSize: 12, fontWeight: 500, color: "var(--text-2)", cursor: "pointer", transition: "all 0.15s" }}
                            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text)"; }}
                            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-2)"; }}
                        >
                            <RefreshCw style={{ width: 13, height: 13 }} />
                            Refresh
                        </button>
                    )}
                </div>

                {/* ── Loading skeleton ── */}
                {loading && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                        {[90, 60, 80, 45, 70, 55, 85, 40].map((w, i) => (
                            <div key={i} style={{ height: i % 3 === 0 ? 20 : 14, borderRadius: 6, background: "var(--bg-3)", width: `${w}%`, animation: "pulse 1.5s ease-in-out infinite", animationDelay: `${i * 0.07}s` }} />
                        ))}
                        <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
                    </div>
                )}

                {/* ── Error ── */}
                {error && (
                    <div style={{ borderRadius: 14, padding: "24px 28px", background: "var(--bg-2)", border: "1px solid var(--border)", textAlign: "center" }}>
                        <AlertCircle style={{ width: 32, height: 32, color: "#e86026", margin: "0 auto 12px" }} />
                        <p style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", margin: "0 0 6px" }}>Could not load README</p>
                        <p style={{ fontSize: 12, color: "var(--text-3)", fontFamily: "monospace", margin: "0 0 16px" }}>{error}</p>
                        <p style={{ fontSize: 12, color: "var(--text-2)", margin: "0 0 16px" }}>
                            Set <code style={{ color: "#e86026", background: "var(--bg-3)", padding: "2px 6px", borderRadius: 4 }}>NEXT_PUBLIC_README_URL</code> in your <code style={{ color: "#e86026", background: "var(--bg-3)", padding: "2px 6px", borderRadius: 4 }}>.env.local</code>
                        </p>
                        <button onClick={fetchReadme} style={{ padding: "8px 18px", borderRadius: 9, background: "#e86026", color: "#fff", border: "none", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
                            Try again
                        </button>
                    </div>
                )}

                {/* ── Placeholder (URL not set) ── */}
                {IS_PLACEHOLDER && !loading && (
                    <div style={{ borderRadius: 14, padding: "32px 28px", background: "var(--bg-2)", border: "1px solid var(--border)", textAlign: "center" }}>
                        <Github style={{ width: 36, height: 36, color: "var(--text-3)", margin: "0 auto 16px" }} />
                        <p style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", margin: "0 0 8px" }}>Connect your README</p>
                        <p style={{ fontSize: 13, color: "var(--text-2)", margin: "0 0 20px", lineHeight: 1.6 }}>
                            Add this to your <code style={{ color: "#e86026", background: "var(--bg-3)", padding: "2px 6px", borderRadius: 4 }}>.env.local</code> file:
                        </p>
                        <div style={{ background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 10, padding: "14px 18px", textAlign: "left", display: "inline-block", fontFamily: "monospace", fontSize: 13, color: "var(--text-2)", marginBottom: 20 }}>
                            <span style={{ color: "var(--text-3)" }}># .env.local</span><br />
                            NEXT_PUBLIC_README_URL=<span style={{ color: "#e86026" }}>https://raw.githubusercontent.com/megashyam/Rezrec/master/README.md</span>
                        </div>
                        <p style={{ fontSize: 12, color: "var(--text-3)", margin: 0 }}>
                            Works with any public GitHub repository
                        </p>
                    </div>
                )}

                {/* ── Content ── */}
                {!loading && !error && !IS_PLACEHOLDER && content && (
                    <div style={{ borderRadius: 16, overflow: "hidden", background: "var(--bg-2)", border: "1px solid var(--border)" }}>
                        {view === "rendered" ? (
                            <div className="prose-content" style={{ padding: "36px 40px" }}>
                                <ReactMarkdown
                                    components={{
                                        h1: ({ node, ...props }) => <h1 {...props} />,
                                        h2: ({ node, ...props }) => <h2 {...props} />,
                                        h3: ({ node, ...props }) => <h3 {...props} />,
                                        h4: ({ node, ...props }) => <h4 {...props} />,
                                        p: ({ node, ...props }) => <p  {...props} />,
                                        ul: ({ node, ...props }) => <ul {...props} />,
                                        ol: ({ node, ...props }) => <ol {...props} />,
                                        li: ({ node, ...props }) => <li {...props} />,
                                        a: ({ node, href, ...props }) => (
                                            <a href={href} target="_blank" rel="noopener noreferrer" {...props} />
                                        ),
                                        code: ({ node, inline, ...props }: any) =>
                                            inline ? <code {...props} /> : <code {...props} />,
                                        pre: ({ node, ...props }) => <pre {...props} />,
                                        blockquote: ({ node, ...props }) => <blockquote {...props} />,
                                        hr: ({ node, ...props }) => <hr  {...props} />,
                                        strong: ({ node, ...props }) => <strong {...props} />,
                                        table: ({ node, ...props }) => (
                                            <div style={{ overflowX: "auto" }}>
                                                <table {...props} />
                                            </div>
                                        ),
                                        th: ({ node, ...props }) => <th {...props} />,
                                        td: ({ node, ...props }) => <td {...props} />,
                                        img: ({ node, src, alt, ...props }) => {
                                            // Resolve relative image paths against the repo root
                                            const srcStr = typeof src === "string" ? src : "";
                                            const resolvedSrc = srcStr.startsWith("http")
                                                ? srcStr
                                                : `https://raw.githubusercontent.com/megashyam/Rezrec/master/${srcStr}`;

                                            return (
                                                <img
                                                    src={resolvedSrc}
                                                    alt={alt || ""}
                                                    style={{
                                                        maxWidth: "100%",
                                                        borderRadius: 8,
                                                        margin: "12px 0",
                                                        border: "1px solid var(--border)",
                                                    }}
                                                    {...props}
                                                />
                                            );
                                        },


                                    }}
                                >
                                    {content}
                                </ReactMarkdown>
                            </div>
                        ) : (
                            <pre style={{
                                padding: "32px 36px",
                                overflowX: "auto",
                                fontSize: 13,
                                lineHeight: 1.7,
                                fontFamily: "var(--font-geist-mono), monospace",
                                color: "var(--text-2)",
                                margin: 0,
                                whiteSpace: "pre-wrap",
                                wordBreak: "break-word",
                            }}>
                                {content}
                            </pre>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
}