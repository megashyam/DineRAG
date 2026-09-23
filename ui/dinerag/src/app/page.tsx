"use client";

import { useState, KeyboardEvent, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  UtensilsCrossed, MapPin, Clock, Sparkles,
  ArrowRight, BookOpen, Star, Navigation, Zap,
  Code,
  Code2,
  Wrench,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { RestaurantCardRow } from "@/components/RestaurantCard";
import { TimingBar, QueryMeta } from "@/components/TimingBar";

const GENERATOR_URL = process.env.NEXT_PUBLIC_GENERATOR_URL;
console.log("GENERATOR_URL:", process.env.NEXT_PUBLIC_GENERATOR_URL);
const MapPanel = dynamic(() => import("@/components/MapPanel"), {
  ssr: false,
  loading: () => (
    <div style={{ width: "100%", height: "100%", background: "var(--bg-3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <p style={{ fontSize: 12, color: "var(--text-3)" }}>Loading map…</p>
    </div>
  ),
});

declare global {
  interface Window { gtag: Function; }
}

// ── Types ──────────────────────────────────────────────────────────────────
type Source = {
  name?: string;
  restaurant?: string;
  address?: string;
  city?: string;
  state?: string;
  text?: string;
  excerpt?: string;
  lat?: number;
  lon?: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  meta?: QueryMeta;
};

// ── Helpers ────────────────────────────────────────────────────────────────
function getSourceName(s: Source): string {
  return (s.name || s.restaurant || "").trim();
}

function extractMentionedSources(text: string, pool: Source[]): Source[] {
  const lower = text.toLowerCase();
  return pool.filter((s) => {
    const fullName = getSourceName(s);
    if (fullName.length <= 3) return false;
    const nameLower = fullName.toLowerCase();
    if (lower.includes(nameLower)) return true;
    const words = nameLower.split(/[\s,&']+/).filter((w) => w.length > 3);
    if (words.length > 0 && words.slice(0, 2).every((w) => lower.includes(w))) return true;
    return false;
  });
}

// ── Component ──────────────────────────────────────────────────────────────
export default function Home() {
  const abortControllerRef = useRef<AbortController | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeSources, setActiveSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);
  const [topK, setTopK] = useState(4);
  const [showMap, setShowMap] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<{
    retriever: "checking" | "warm" | "cold";
    generator: "checking" | "warm" | "cold";
  } | null>({ retriever: "checking", generator: "checking" });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const allSourcesRef = useRef<Source[]>([]);
  const [currentMeta, setCurrentMeta] = useState<QueryMeta | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [showPerf, setShowPerf] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [input]);

  useEffect(() => {
    fetch("/api/geo")
      .then(r => r.json())
      .then(geo => {
        if (typeof window !== "undefined" && window.gtag) {
          window.gtag("event", "user_visit", {
            city: geo.city,
            region: geo.region,
            country: geo.country,
          });
        }
      });
  }, []);

  useEffect(() => {
    const RETRIEVER_URL = process.env.NEXT_PUBLIC_RETRIEVER_URL;

    const check = async (url: string) => {
      try {
        const res = await fetch(`/api/health?url=${encodeURIComponent(url!)}`);
        const data = await res.json();
        return data.ok ? "warm" : "cold";
      } catch {
        return "cold";
      }
    };



    let interval: NodeJS.Timeout;

    const poll = async () => {
      const [retriever, generator] = await Promise.all([
        check(RETRIEVER_URL!),
        check(GENERATOR_URL!),
      ]);

      setServiceStatus({ retriever, generator } as any);

      // Once both warm — stop polling and hide after 3s
      if (retriever === "warm" && generator === "warm") {
        clearInterval(interval);
        setTimeout(() => setServiceStatus(null), 3000);
      }
    };

    poll();
    interval = setInterval(poll, 10000);

    // Hard stop after 3 minutes regardless
    const stop = setTimeout(() => {
      clearInterval(interval);
      setServiceStatus(null);
    }, 180000);

    return () => {
      clearInterval(interval);
      clearTimeout(stop);
    };
  }, []);
  async function send(queryOverride?: string) {

    const textToSend = queryOverride || input;
    if (!textToSend.trim() || busy) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setMessages((prev) => [...prev, { role: "user", content: textToSend }]);
    setBusy(true);
    setInput("");
    setActiveSources([]);
    allSourcesRef.current = [];
    setCurrentMeta(null);
    setShowMap(false);
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    let assistantText = "";
    let updatePending = false;
    console.log("GENERATOR_URL:", process.env.NEXT_PUBLIC_GENERATOR_URL);

    try {
      const res = await fetch(`${GENERATOR_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: textToSend, city: null, top_k: topK, }),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok || !res.body) throw new Error(`Server error: ${res.status}`);



      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);

            // ── Meta event (timing bar) ──────────────────────────────────────────
            if (json.type === "meta") {
              setCurrentMeta(json.data as QueryMeta);
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy.length - 1;
                if (copy[last].role === "assistant") {
                  copy[last] = { ...copy[last], meta: json.data };
                }
                return copy;
              });
            }
            else if (json.type === "sources") {
              allSourcesRef.current = json.data as Source[];
              setActiveSources(json.data as Source[]);
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy.length - 1;
                if (copy[last].role === "assistant") {
                  copy[last] = { ...copy[last], sources: json.data };
                }
                return copy;
              });
            }
            else if (json.type === "token") {
              assistantText += json.data;
              const captured = assistantText;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy.length - 1;
                if (copy[last].role === "assistant") {
                  copy[last] = { ...copy[last], content: captured };
                }
                return copy;
              });

            }
            else if (json.type === "error") {
              console.error("[backend error]", json.data);
              assistantText += `\n\n*[Error: ${json.data}]*`;
            }
          } catch (e) {
            console.error("[json parse error]", e);
          }
        }
      }
    } catch (err) {

      if ((err as Error).name === "AbortError") {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy.length - 1;
          if (copy[last].role === "assistant" && copy[last].content === "") {
            copy[last] = { ...copy[last], content: "_Response stopped._" };
          }
          return copy;
        });
        setBusy(false);
        return;
      }
      console.error("[fetch error]", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "❌ Could not connect to the food guide API." },
      ]);
    } finally {
      if (assistantText) {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy.length - 1;
          if (copy[last].role === "assistant") {
            copy[last] = { ...copy[last], content: assistantText };
          }
          return copy;
        });
      }
      setBusy(false);
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  // ── Hero ──────────────────────────────────────────────────────────────────
  const renderHero = () => (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", position: "relative", overflow: "auto", padding: "16px 16px 8px" }}>

      {/* Animated blobs */}
      <div style={{ position: "absolute", inset: 0, overflow: "hidden", pointerEvents: "none" }}>
        <div className="blob-1" style={{ position: "absolute", top: "20%", left: "20%", width: 340, height: 340, borderRadius: "50%", background: "#e86026", opacity: 0.06, filter: "blur(70px)" }} />
        <div className="blob-2" style={{ position: "absolute", bottom: "20%", right: "25%", width: 400, height: 400, borderRadius: "50%", background: "#9333ea", opacity: 0.04, filter: "blur(90px)" }} />
        <div className="blob-3" style={{ position: "absolute", top: "50%", right: "15%", width: 280, height: 280, borderRadius: "50%", background: "#f59e0b", opacity: 0.04, filter: "blur(80px)" }} />
      </div>

      {/* Dot grid */}
      <div style={{
        position: "absolute", inset: 0, pointerEvents: "none",
        backgroundImage: "radial-gradient(circle, var(--border-2) 1px, transparent 1px)",
        backgroundSize: "28px 28px", opacity: 0.6,
      }} />

      <div style={{ position: "relative", zIndex: 10, maxWidth: 600, width: "100%", textAlign: "center" }}>

        {/* Status badge */}
        <div className="fade-up" style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "6px 14px", borderRadius: 99, background: "var(--bg-2)", border: "1px solid var(--border-2)", fontSize: 11, fontWeight: 600, color: "var(--text-2)", marginBottom: 24 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#22c55e", display: "inline-block", animation: "pulse 2s ease-in-out infinite" }} />
          Yelp RAG · Built from Scratch
        </div>

        {/* Heading */}
        <h1 className="fade-up-1" style={{ fontSize: "clamp(2rem, 5vw, 3.6rem)", fontWeight: 800, lineHeight: 1.1, letterSpacing: "-0.03em", margin: "0 0 18px", color: "var(--text)" }}>
          Find your next<br />
          <span style={{ color: "var(--accent)" }}>perfect meal.</span>
        </h1>

        <p className="fade-up-2" style={{ fontSize: 17, lineHeight: 1.75, color: "var(--text-2)", margin: "0 0 30px", maxWidth: 440, marginLeft: "auto", marginRight: "auto" }}>
          Ask anything about local restaurants. Real Yelp reviews, real recommendations
          built entirely from scratch without LangChain, LlamaIndex, or any RAG framework.
        </p>


        {/* Coverage note */}
        <div style={{
          margin: "0 auto 20px",
          padding: "5px 10px",
          borderRadius: 10,
          // background: "var(--bg-4)",
          // border: "1px solid var(--border)",
          fontSize: 11,
          color: "var(--text-2)",
          textAlign: "center",
          lineHeight: 1.8,
          maxWidth: 480,
          marginLeft: "auto",
          marginRight: "auto",
        }}>
          <span style={{ fontWeight: 600, color: "var(--text-2)" }}>📍 The Yelp Dataset only covers</span>
          {" · "}
          Philadelphia · Nashville · Tampa · New Orleans · Indianapolis · Tucson · Reno · Boise · Santa Barbara · Edmonton · Saint Louis · Wilmington · Belleville
          <br />
          <span style={{ fontSize: 11, fontStyle: "italic" }}>
            + surrounding suburbs across PA, NJ, FL, TN, LA, IN, AZ, NV, ID, CA, IL, MO, DE & Alberta
          </span>
        </div>


        {/* Feature pills */}
        <div className="fade-up-3" style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8, marginBottom: 36 }}>
          {[
            { icon: Star, label: "Real Yelp data" },
            { icon: Navigation, label: "Location-aware" },
            { icon: Code2, label: "Zero RAG frameworks" },
            { icon: Wrench, label: "Hand-rolled pipeline" },
          ].map(({ icon: Icon, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 13px", borderRadius: 99, background: "var(--bg-2)", border: "1px solid var(--border)", fontSize: 12, fontWeight: 500, color: "var(--text-2)" }}>
              <Icon style={{ width: 13, height: 13, color: "#e86026", flexShrink: 0 }} />
              {label}
            </div>
          ))}
        </div>




        {/* Pipeline showcase */}
        <div className="fade-up-3" style={{
          width: "100%", maxWidth: 480,
          margin: "0 auto 28px",
          background: "var(--bg-2)",
          border: "1px solid var(--border)",
          borderRadius: 16, padding: "16px 20px",
        }}>
          <p style={{ fontSize: 11, fontWeight: 700, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase", margin: "0 0 12px" }}>
            Zero abstractions · Built from scratch
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[
              { step: "01", label: "Raw Yelp JSON", detail: "~10GB, multi-file, messy" },
              { step: "02", label: "Clean + Score + Sample", detail: "Custom restaurant scoring, balanced sentiment" },
              { step: "03", label: "E5-Tokenizer Chunking", detail: "Typed chunks — profile, positive, negative" },
              { step: "04", label: "E5-large-v2 Embeddings", detail: "1024-dim dense vectors + full-corpus BM25 index" },
              { step: "05", label: "Qdrant Ingestion", detail: "UUID5 stable IDs, generator-based batching" },
              { step: "06", label: "Intent Classification", detail: "llama-3.1-8b-instant via Groq routes before retrieval" },
              { step: "07", label: "Hybrid Search + RRF", detail: "Dense + full-corpus sparse fusion, no framework" },
              { step: "08", label: "CrossEncoder Rerank", detail: "ms-marco-MiniLM-L-6-v2, own 512-token tokenizer" },
              { step: "09", label: "Qwen 3 32B via Groq (For production)", detail: "Streaming , token-by-token" },
            ].map(({ step, label, detail }, i, arr) => (
              <div key={step} style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                {/* Step number + connector */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: "50%",
                    background: "var(--bg-3)", border: "1px solid var(--border-2)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 9, fontWeight: 800, color: "#e86026", fontFamily: "monospace",
                  }}>
                    {step}
                  </div>
                  {i < arr.length - 1 && (
                    <div style={{ width: 1, height: 8, background: "var(--border)", marginTop: 2 }} />
                  )}
                </div>
                {/* Label + detail */}
                <div style={{ paddingTop: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{label}</span>
                  <span style={{ fontSize: 11, color: "var(--text-3)", marginLeft: 6 }}>{detail}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Performance optimizations — collapsible */}



        <div style={{ width: "100%", maxWidth: 480, margin: "0 auto 28px" }}>
          <button
            onClick={() => setShowPerf(v => !v)}
            style={{
              width: "100%",
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "10px 16px", borderRadius: showPerf ? "12px 12px 0 0" : 12,
              background: "var(--bg-2)", border: "1px solid var(--border)",
              color: "var(--text-2)", fontSize: 12, fontWeight: 700,
              cursor: "pointer", transition: "all 0.2s",
              letterSpacing: "0.06em", textTransform: "uppercase",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ color: "#e86026" }}>⚡</span>
              Performance Optimizations
            </span>
            <span style={{ fontSize: 16, color: "var(--text-3)", transition: "transform 0.2s", transform: showPerf ? "rotate(180deg)" : "rotate(0deg)" }}>
              ›
            </span>
          </button>

          {showPerf && (
            <div style={{
              background: "var(--bg-2)", border: "1px solid var(--border)",
              borderTop: "none", borderRadius: "0 0 12px 12px",
              padding: "12px 16px", display: "flex", flexDirection: "column", gap: 6,
            }}>
              {[
                { layer: "Data", items: ["orjson 3–5× faster than stdlib json", "Filter cascade: O(1) set → date string → word count", "df.melt() + df.explode() — C-level vectorization", "Generator-based Qdrant ingestion — flat RAM"] },
                { layer: "Chunking", items: ["Long single reviews truncated", "Single-pass progress_apply"] },
                { layer: "Retrieval", items: ["Full-corpus BM25 fused with dense search", "Reranker truncates via its own 512-token tokenizer", "spaCy NER geo-filter cuts Qdrant search space"] },
                { layer: "Deployment", items: ["Modal Volume — models cached, not re-downloaded", "keep_warm=1 — one container always hot", "Qdrant co-located in same GCP region as Modal", "6-minute query cache — repeated queries skip retrieval"] },
                { layer: "Frontend", items: ["setTimeout(fn,30) — token updates batched to ~60fps", "Leaflet lazy-loaded (ssr:false) — no SSR crash", "Health check via API route — no CORS on Modal calls"] },
              ].map(({ layer, items }) => (
                <div key={layer}>
                  <div style={{ fontSize: 10, fontWeight: 800, color: "#e86026", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>
                    {layer}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    {items.map(item => (
                      <div key={item} style={{ display: "flex", alignItems: "flex-start", gap: 6, fontSize: 11, color: "var(--text-2)" }}>
                        <span style={{ color: "var(--text-3)", marginTop: 1, flexShrink: 0 }}>·</span>
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        {/* Query chips */}
        <div className="fade-up-4" style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10 }}>
          {[
            { label: "Best pizzerias in Santa Barbara", icon: UtensilsCrossed },
            { label: "Family friendly sushi restaurants", icon: MapPin },
            { label: "Best seafood places with waterfront views", icon: Clock },
            { label: "Bars with live music and cocktails Philadelphia", icon: Star },
          ].map((chip) => (
            <button
              key={chip.label}
              onClick={() => send(chip.label)}
              style={{
                display: "flex", alignItems: "center", gap: 9,
                padding: "10px 16px",
                borderRadius: 99,
                background: "var(--bg-2)",
                border: "1px solid var(--border-2)",
                fontSize: 13, fontWeight: 500, color: "var(--text-2)",
                cursor: "pointer",
                transition: "all 0.18s ease",
              }}
              onMouseEnter={(e) => {
                const b = e.currentTarget;
                b.style.borderColor = "#e86026";
                b.style.color = "var(--text)";
                b.style.background = "var(--bg-3)";
                b.style.transform = "scale(1.03)";
              }}
              onMouseLeave={(e) => {
                const b = e.currentTarget;
                b.style.borderColor = "var(--border-2)";
                b.style.color = "var(--text-2)";
                b.style.background = "var(--bg-2)";
                b.style.transform = "scale(1)";
              }}
            >
              <chip.icon style={{ width: 14, height: 14, color: "#e86026", flexShrink: 0 }} />
              {chip.label}
            </button>
          ))}
        </div>

      </div>
    </div>
  );

  // ── Root ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--bg)" }}>
      {/* Service status banner */}
      {serviceStatus && (
        <div style={{
          position: "fixed", bottom: 80, left: "50%", transform: "translateX(-50%)",
          zIndex: 500,
          display: "flex",
          flexDirection: "column",    // ← stack vertically
          alignItems: "center",
          gap: 6,
          padding: "10px 16px",
          borderRadius: 12,
          background: "var(--bg-2)", border: "1px solid var(--border)",
          boxShadow: "0 4px 24px rgba(0,0,0,0.12)",
          fontSize: 12, fontWeight: 500, color: "var(--text-2)",
          width: "calc(100vw - 32px)",   // ← full width minus margins
          maxWidth: 360,                  // ← cap on desktop
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", justifyContent: "center" }}>
            <span style={{ fontWeight: 700, color: "var(--text)" }}>Service Status</span>
            {(serviceStatus.retriever === "warm" && serviceStatus.generator === "warm") && (
              <span style={{ color: "#22c55e" }}>· ready</span>
            )}
          </div>

          <div style={{ display: "flex", gap: 16, justifyContent: "center" }}>
            {(["retriever", "generator"] as const).map((svc) => {
              const status = serviceStatus[svc];
              const color = status === "warm" ? "#22c55e" : status === "checking" ? "#f59e0b" : "#e86026";
              const label = status === "checking" ? "starting…" : status === "warm" ? "warm" : "cold start";
              return (
                <div key={svc} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{
                    width: 7, height: 7, borderRadius: "50%", background: color,
                    display: "inline-block",
                    animation: status === "checking" ? "pulse 1.5s ease-in-out infinite" : "none",
                  }} />
                  <span style={{ textTransform: "capitalize" }}>{svc}</span>
                  <span style={{ color: "var(--text-3)" }}>·</span>
                  <span style={{ color }}>{label}</span>
                </div>
              );
            })}
          </div>

          {(serviceStatus.retriever === "cold" || serviceStatus.generator === "cold") && (
            <span style={{ color: "var(--text-3)", fontStyle: "italic", textAlign: "center" }}>
              Cold start detected · first query may take ~20s
            </span>
          )}
        </div>
      )}
      {/* Main area */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex" }}>
        {messages.length === 0 ? renderHero() : renderChatWithMap()}
      </div>

      {/* Input footer */}
      <footer style={{
        flexShrink: 0,
        padding: "12px 16px",
        paddingBottom: "max(12px, env(safe-area-inset-bottom, 12px))",
        background: "var(--bg)",
        borderTop: "1px solid var(--border)",
        zIndex: 30,
      }}>
        <div style={{
          maxWidth: 720,
          margin: "0 auto",
          position: "relative",
        }}>
          {/* Input box */}
          <div style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 8,
            background: "var(--bg-2)",
            border: "1px solid var(--border)",
            borderRadius: 16,
            padding: "8px 8px 8px 14px",
            transition: "border-color 0.2s",
          }}>
            <textarea
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
              onKeyDown={handleKeyDown}
              disabled={busy}
              placeholder="Find the best spicy wings, romantic Italian, late night tacos..."
              rows={1}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                resize: "none",
                color: "var(--text)",
                fontSize: 15,
                lineHeight: 1.5,
                padding: "4px 0",
                maxHeight: 120,
                overflowY: "auto",
                fontFamily: "inherit",
              }}
            />
            <button
              onClick={() => send()}
              disabled={busy || !input.trim()}
              style={{
                width: 36, height: 36, borderRadius: 10,
                background: !input.trim() || busy ? "var(--bg-3)" : "#e86026",
                color: !input.trim() || busy ? "var(--text-3)" : "#fff",
                border: "none", cursor: !input.trim() || busy ? "default" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, transition: "all 0.2s",
              }}
            >
              {busy ? (
                <span style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid currentColor", borderTopColor: "transparent", animation: "spin 0.8s linear infinite", display: "inline-block" }} />
              ) : (
                <ArrowRight style={{ width: 16, height: 16 }} />
              )}
            </button>

            {busy && (
              <button
                onClick={() => {
                  abortControllerRef.current?.abort();
                  setBusy(false);
                  setMessages((prev) => {
                    const copy = [...prev];
                    const last = copy.length - 1;
                    if (copy[last].role === "assistant" && copy[last].content === "") {
                      copy[last] = { ...copy[last], content: "_Response stopped._" };
                    }
                    return copy;
                  });
                }}
                style={{
                  width: 36, height: 36, borderRadius: 10,
                  background: "var(--bg-3)", border: "1px solid var(--border)",
                  color: "var(--text-2)", cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0, transition: "all 0.2s",
                }}
              >
                <span style={{ width: 12, height: 12, background: "currentColor", borderRadius: 2, display: "inline-block" }} />
              </button>
            )}
          </div>

          {/* Controls row */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            marginTop: 8,
          }}>
            {/* top_k stepper */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 11, color: "var(--text-3)", fontWeight: 500 }}>Results:</span>
              <button
                onClick={() => setTopK((v) => Math.max(3, v - 1))}
                disabled={topK <= 3}
                style={{
                  width: 20, height: 20, borderRadius: 6,
                  border: "1px solid var(--border)", background: "var(--bg-3)",
                  color: "var(--text-2)", fontSize: 13,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: topK <= 3 ? "default" : "pointer",
                  opacity: topK <= 3 ? 0.4 : 1, lineHeight: 1, padding: 0,
                }}
              >−</button>
              <span style={{
                fontSize: 12, fontWeight: 700, color: "var(--text-2)",
                minWidth: 18, textAlign: "center", fontFamily: "monospace",
              }}>{topK}</span>
              <button
                onClick={() => setTopK((v) => Math.min(15, v + 1))}
                disabled={topK >= 15}
                style={{
                  width: 20, height: 20, borderRadius: 6,
                  border: "1px solid var(--border)", background: "var(--bg-3)",
                  color: "var(--text-2)", fontSize: 13,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: topK >= 15 ? "default" : "pointer",
                  opacity: topK >= 15 ? 0.4 : 1, lineHeight: 1, padding: 0,
                }}
              >+</button>
            </div>

            <span style={{ fontSize: 11, color: "var(--text-3)" }}>
              Enter to send · Shift+Enter for new line
            </span>
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes ping { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.6);opacity:0.4} }
        @media (min-width:1024px) {
          .map-col { display: block !important; }
        }
      `}</style>
    </div>
  );

  // Separate function to handle the map column visibility via className
  // Replace your entire renderChatWithMap function with this.
  // Key fix: no inline style={{ display }} fighting with Tailwind classes.
  // Mobile toggle is handled purely via CSS classes.

  function renderChatWithMap() {
    const hasMap = activeSources.length > 0;

    return (
      <div style={{ display: "flex", flex: 1, height: "100%", overflow: "hidden", flexDirection: "column" }}>

        {/* Mobile map toggle — only visible on small screens when there are results */}
        {hasMap && (
          <div className="flex lg:hidden" style={{
            alignItems: "center",
            justifyContent: "center",
            padding: "6px 16px",
            borderBottom: "1px solid var(--border)",
            flexShrink: 0,
          }}>
            <button
              onClick={() => setShowMap((v) => !v)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 16px",
                borderRadius: 20,
                border: "1px solid var(--border)",
                background: showMap ? "#e86026" : "var(--bg-2)",
                color: showMap ? "#fff" : "var(--text-2)",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              <span>🗺</span>
              {showMap ? "Hide Map" : `Show Map (${activeSources.length})`}
            </button>
          </div>
        )}

        {/* Content row */}
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

          {/* Chat panel */}
          <div
            className={`chat-panel ${hasMap ? "chat-panel-with-map" : ""} ${showMap ? "hidden lg:flex" : "flex"}`}
            style={{
              flexDirection: "column",
              height: "100%",
              overflow: "hidden",
              transition: "flex 0.4s ease",
            }}
          >
            <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 20 }}>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className="fade-in"
                  style={{ display: "flex", gap: 10, justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}
                >
                  {msg.role === "assistant" && (
                    <div style={{ width: 28, height: 28, borderRadius: "50%", flexShrink: 0, marginTop: 2, background: "var(--bg-3)", border: "1px solid var(--border-2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Sparkles style={{ width: 13, height: 13, color: "#e86026" }} />
                    </div>
                  )}

                  <div style={{
                    maxWidth: "96%",
                    padding: "12px 16px",
                    borderRadius: msg.role === "user" ? "18px 6px 18px 18px" : "6px 18px 18px 18px",
                    ...(msg.role === "user"
                      ? { background: "var(--bg-3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 14 }
                      : { color: "var(--text-2)", fontSize: 14, lineHeight: 1.75 }
                    ),
                  }}>
                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div style={{ marginBottom: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {msg.sources.slice(0, 4).map((s, i) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 8px", borderRadius: 6, background: "var(--bg-3)", border: "1px solid var(--border)", fontSize: 10, fontWeight: 500, color: "var(--text-3)" }}>
                            <BookOpen style={{ width: 10, height: 10 }} />
                            <span style={{ maxWidth: 100, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {getSourceName(s)}
                            </span>
                          </div>
                        ))}
                        {msg.sources.length > 4 && (
                          <div style={{ padding: "3px 6px", fontSize: 10, color: "var(--text-3)" }}>
                            +{msg.sources.length - 4} more
                          </div>
                        )}
                      </div>
                    )}

                    {msg.role === "assistant" && msg.content === "" ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ position: "relative", display: "inline-flex", width: 8, height: 8 }}>
                          <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#e86026", opacity: 0.75, animation: "ping 1s ease-in-out infinite" }} />
                          <span style={{ position: "relative", width: 8, height: 8, borderRadius: "50%", background: "#e86026", display: "inline-flex" }} />
                        </span>
                        <span style={{ fontSize: 13, color: "var(--text-3)" }}>Searching restaurants…</span>
                      </div>
                    ) : (
                      <>
                        {msg.meta && <TimingBar meta={msg.meta} />}
                        {msg.sources && msg.sources.length > 0 && (
                          <RestaurantCardRow sources={msg.sources} />
                        )}
                        <ReactMarkdown
                          components={{
                            strong: ({ node, ...props }) => <span style={{ color: "#e86026", fontWeight: 600 }} {...props} />,
                            ul: ({ node, ...props }) => <ul style={{ paddingLeft: "1.25rem", margin: "6px 0" }} {...props} />,
                            li: ({ node, ...props }) => <li style={{ marginBottom: 4 }} {...props} />,
                            p: ({ node, ...props }) => <p style={{ margin: "0 0 10px", lineHeight: 1.75 }} {...props} />,
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} style={{ height: 8 }} />
            </div>
          </div>

          {/* Map panel */}
          {/* Map panel — side panel on desktop, fullscreen overlay on mobile */}
          {hasMap && (
            <>
              {/* Desktop: side panel */}
              <div
                className="map-col hidden lg:flex"
                style={{ flex: "0 0 45%", height: "100%", borderLeft: "1px solid var(--border)", position: "relative" }}
              >
                <MapPanel sources={activeSources} />
                <div style={{ position: "absolute", top: 12, left: 12, zIndex: 400, fontSize: 11, fontWeight: 700, padding: "4px 10px", borderRadius: 99, background: "var(--bg-2)", border: "1px solid var(--border-2)", color: "var(--text-2)" }}>
                  {activeSources.length} spots found
                </div>
              </div>

              {/* Mobile: fullscreen overlay */}
              {showMap && (
                <div className="lg:hidden" style={{ position: "fixed", inset: 0, zIndex: 200 }}>
                  <MapPanel sources={activeSources} />
                  <div style={{ position: "absolute", top: 12, left: 12, zIndex: 400, fontSize: 11, fontWeight: 700, padding: "4px 10px", borderRadius: 99, background: "var(--bg-2)", border: "1px solid var(--border-2)", color: "var(--text-2)" }}>
                    {activeSources.length} spots found
                  </div>
                  <button
                    onClick={() => setShowMap(false)}


                    style={{
                      position: "absolute", top: 12, right: 12, zIndex: 1000,
                      width: 36, height: 36, borderRadius: "50%",
                      background: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)",
                      color: "#fff", fontSize: 18, cursor: "pointer",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}
                  >✕</button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    );
  }
}



