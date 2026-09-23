"use client";

import {
    MapContainer,
    TileLayer,
    Marker,
    Popup,
    useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";

// ── Types ──────────────────────────────────────────────────────────────────
type Source = {
    name?: string;
    restaurant?: string;
    address?: string;
    city?: string;
    state?: string;
    excerpt?: string;
    lat?: number;
    lon?: number;
};

type ValidMarker = Source & { lat: number; lon: number };

// ── Marker icon factory ────────────────────────────────────────────────────
function makeIcon(num: number, hovered: boolean): L.DivIcon {
    const s = hovered ? 44 : 34;
    const half = s / 2;
    return L.divIcon({
        className: "",
        html: `<div style="
      width:${s}px;height:${s}px;
      background:${hovered ? "#e86026" : "#111113"};
      border:2px solid ${hovered ? "#ffb38a" : "#e86026"};
      border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      color:${hovered ? "#fff" : "#e86026"};
      font-weight:700;font-size:13px;font-family:system-ui,sans-serif;
      box-shadow:${hovered
                ? "0 0 0 6px rgba(232,96,38,0.25),0 4px 12px rgba(0,0,0,0.7)"
                : "0 2px 8px rgba(0,0,0,0.6)"};
      transition:all 0.18s ease;
      cursor:pointer;
      position:relative;
    ">${num}${hovered ? `<div style="
        position:absolute;inset:-6px;border-radius:50%;
        border:1.5px solid rgba(232,96,38,0.4);
        animation:ring 1.2s ease-out infinite;
      "></div>` : ""
            }</div>`,
        iconSize: [s, s],
        iconAnchor: [half, half],
        popupAnchor: [0, -half - 6],
    });
}

// ── Auto-fit bounds — only when source set changes, never on hover ─────────
function MapBounds({ markers }: { markers: ValidMarker[] }) {
    const map = useMap();
    // Stable key: sorted joined coords — only changes when actual sources change
    const boundsKey = markers.map((m) => `${m.lat},${m.lon}`).sort().join("|");

    useEffect(() => {
        if (markers.length === 0) return;
        const bounds = L.latLngBounds(markers.map((m) => [m.lat, m.lon]));
        map.fitBounds(bounds, { padding: [60, 260], maxZoom: 14 });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [boundsKey]); // ← intentionally NOT [markers] to prevent hover re-zoom

    return null;
}
function DetailPanel({
    source,
    num,
    onClose,
}: {
    source: ValidMarker;
    num: number;
    onClose: () => void;
}) {
    const name = source.name || source.restaurant || "Restaurant";

    const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(
        `${name} ${source.address || ""}`
    )}`;
    const yelpUrl = `https://www.yelp.com/search?find_desc=${encodeURIComponent(
        name
    )}&find_loc=${encodeURIComponent(
        `${source.city || ""} ${source.state || ""}`
    )}`;

    const btnBase: React.CSSProperties = {
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        padding: "8px 12px",
        background: "#1a1a1c",
        border: "1px solid #2a2a2a",
        borderRadius: 8,
        color: "#ccc",
        fontSize: 11,
        fontWeight: 600,
        textDecoration: "none",
        transition: "all 0.15s",
    };

    return (
        <div style={{
            position: "absolute",
            bottom: 12, left: 12, right: 12,
            zIndex: 600,
            background: "rgba(8,8,10,0.96)",
            border: "1px solid #2a2a2a",
            borderRadius: 16,
            backdropFilter: "blur(12px)",
            overflow: "hidden",
            animation: "slideUp 0.2s ease",
        }}>

            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderBottom: "1px solid #1e1e1e" }}>
                <div style={{ width: 30, height: 30, borderRadius: "50%", background: "#e86026", color: "#fff", fontSize: 13, fontWeight: 700, fontFamily: "system-ui,sans-serif", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    {num}
                </div>
                <div style={{ flex: 1, overflow: "hidden" }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {name}
                    </div>
                    {(source.address || source.city) && (
                        <div style={{ fontSize: 11, color: "#666", marginTop: 1 }}>
                            {[source.address, source.city, source.state].filter(Boolean).join(", ")}
                        </div>
                    )}
                </div>
                <button
                    onClick={onClose}
                    onMouseEnter={(e) => { e.currentTarget.style.color = "#fff"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = "#555"; }}
                    style={{ background: "none", border: "none", color: "#555", cursor: "pointer", fontSize: 18, lineHeight: "1", padding: "4px 6px", borderRadius: 6 }}>
                    ✕
                </button>
            </div>

            {/* Review excerpt */}
            {source.excerpt && (
                <div style={{ padding: "10px 14px", borderBottom: "1px solid #1a1a1a" }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#e86026", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                        From reviews
                    </div>
                    <div style={{ fontSize: 12, color: "#888", lineHeight: 1.6, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                        &ldquo;{source.excerpt}&rdquo;
                    </div>
                </div>
            )}

            {/* Action buttons */}

            <div style={{ display: "flex", gap: 8, padding: "10px 14px" }}>
                <a
                    href={gmapsUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={btnBase}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "#222"; e.currentTarget.style.color = "#fff"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "#1a1a1c"; e.currentTarget.style.color = "#ccc"; }}>
                    📍 Google Maps
                </a>

                <a
                    href={yelpUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={btnBase}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "#c41200"; e.currentTarget.style.color = "#fff"; e.currentTarget.style.borderColor = "#c41200"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "#1a1a1c"; e.currentTarget.style.color = "#ccc"; e.currentTarget.style.borderColor = "#2a2a2a"; }}>
                    ⭐ Yelp
                </a>
            </div >

        </div >
    );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function MapPanel({ sources }: { sources: Source[] }) {
    // Memoize so reference only changes when sources prop changes,
    // NOT when hover state changes — this is what prevents re-zoom on hover
    const markers = useMemo(
        () => sources.filter((s): s is ValidMarker => Boolean(s.lat && s.lon)),
        [sources]
    );

    const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
    const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
    const apiKey = process.env.NEXT_PUBLIC_STADIA_API_KEY;

    const markerRefs = useRef<(L.Marker | null)[]>([]);
    const listRefs = useRef<(HTMLDivElement | null)[]>([]);
    const hoveredRef = useRef<number | null>(null);

    const defaultCenter: [number, number] = [39.5, -98.35];

    const label = (s: Source) => s.name || s.restaurant || "Restaurant";

    // Imperative icon swap — no React re-render for marker icons
    const hoverMarker = useCallback((idx: number | null) => {
        const prev = hoveredRef.current;
        if (prev !== null && markerRefs.current[prev]) {
            markerRefs.current[prev]!.setIcon(makeIcon(prev + 1, false));
        }
        if (idx !== null && markerRefs.current[idx]) {
            markerRefs.current[idx]!.setIcon(makeIcon(idx + 1, true));
            listRefs.current[idx]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
        hoveredRef.current = idx;
        setHoveredIdx(idx);
    }, []);

    const selectMarker = useCallback((idx: number) => {
        setSelectedIdx((prev) => (prev === idx ? null : idx));
    }, []);

    return (
        <div style={{ width: "100%", height: "100%", position: "relative", background: "#0f0f11" }}>

            <style>{`
        @keyframes ring {
          0%   { transform:scale(1);   opacity:0.8; }
          100% { transform:scale(1.8); opacity:0;   }
        }
        @keyframes slideUp {
          from { transform:translateY(12px); opacity:0; }
          to   { transform:translateY(0);    opacity:1; }
        }
        .leaflet-popup-content-wrapper {
          background:rgba(10,10,12,0.97) !important;
          border:1px solid #2a2a2a !important;
          border-radius:12px !important;
          box-shadow:0 12px 32px rgba(0,0,0,0.7) !important;
          padding:0 !important; overflow:hidden;
        }
        .leaflet-popup-tip-container { display:none; }
        .leaflet-popup-content { margin:0 !important; }
        .leaflet-popup-close-button {
          color:#555 !important; font-size:18px !important;
          top:8px !important; right:10px !important; padding:0 !important;
        }
        .leaflet-popup-close-button:hover { color:#e86026 !important; }
        .leaflet-control-attribution {
          background:rgba(0,0,0,0.6) !important;
          color:#444 !important; font-size:9px !important;
        }
        .leaflet-control-attribution a { color:#666 !important; }
        .leaflet-bar a {
          background:#1a1a1c !important; color:#aaa !important;
          border-color:#2a2a2a !important;
        }
        .leaflet-bar a:hover { background:#222 !important; color:#fff !important; }
      `}</style>

            {/* ── Restaurant list ── */}
            <div style={{
                position: "absolute", top: 12, right: 12,
                zIndex: 500,
                width: 220,
                maxHeight: selectedIdx !== null ? "calc(55% - 24px)" : "calc(100% - 24px)",
                display: "flex", flexDirection: "column",
                background: "rgba(8,8,10,0.88)",
                border: "1px solid #232323",
                borderRadius: 14,
                overflow: "hidden",
                backdropFilter: "blur(10px)",
                transition: "max-height 0.3s ease",
            }}>
                <div style={{
                    padding: "10px 14px",
                    borderBottom: "1px solid #1e1e1e",
                    fontSize: 10, fontWeight: 700,
                    letterSpacing: "0.1em", color: "#555",
                    textTransform: "uppercase", flexShrink: 0,
                }}>
                    {markers.length} restaurants found
                </div>

                <div style={{
                    overflowY: "auto", flexGrow: 1,
                    scrollbarWidth: "thin",
                    scrollbarColor: "#2a2a2a transparent",
                }}>
                    {markers.map((s, i) => {
                        const active = hoveredIdx === i;
                        const selected = selectedIdx === i;
                        return (
                            <div
                                key={i}
                                ref={(el) => { listRefs.current[i] = el; }}
                                onMouseEnter={() => hoverMarker(i)}
                                onMouseLeave={() => hoverMarker(null)}
                                onClick={() => selectMarker(i)}
                                style={{
                                    display: "flex", alignItems: "center", gap: 10,
                                    padding: "9px 12px",
                                    cursor: "pointer",
                                    borderLeft: selected
                                        ? "2px solid #ffb38a"
                                        : active
                                            ? "2px solid #e86026"
                                            : "2px solid transparent",
                                    background: selected
                                        ? "rgba(232,96,38,0.14)"
                                        : active
                                            ? "rgba(232,96,38,0.08)"
                                            : "transparent",
                                    transition: "all 0.15s ease",
                                }}
                            >
                                <div style={{
                                    width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
                                    background: active || selected ? "#e86026" : "#161618",
                                    border: `1.5px solid ${active || selected ? "#e86026" : "#333"}`,
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: 11, fontWeight: 700,
                                    fontFamily: "system-ui,sans-serif",
                                    color: active || selected ? "#fff" : "#e86026",
                                    transition: "all 0.15s ease",
                                }}>
                                    {i + 1}
                                </div>
                                <div style={{ overflow: "hidden" }}>
                                    <div style={{
                                        fontSize: 12, fontWeight: 600,
                                        color: active || selected ? "#fff" : "#bbb",
                                        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                                        transition: "color 0.15s",
                                        lineHeight: 1.3,
                                    }}>
                                        {label(s)}
                                    </div>
                                    {s.address && (
                                        <div style={{
                                            fontSize: 10, color: "#555", marginTop: 2,
                                            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                                        }}>
                                            {s.address}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* ── Detail panel (shown on click) ── */}
            {selectedIdx !== null && markers[selectedIdx] && (
                <DetailPanel
                    source={markers[selectedIdx]}
                    num={selectedIdx + 1}
                    onClose={() => setSelectedIdx(null)}
                />
            )}

            {/* ── Map ── */}
            <MapContainer
                center={defaultCenter}
                zoom={4}
                style={{ height: "100%", width: "100%" }}
                zoomControl
            >
                <TileLayer
                    attribution='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url={`https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png?api_key=${apiKey}`}
                    maxZoom={20}
                />

                {markers.map((s, i) => (
                    <Marker
                        key={i}
                        position={[s.lat, s.lon]}
                        icon={makeIcon(i + 1, false)}
                        ref={(el) => { markerRefs.current[i] = el; }}
                        eventHandlers={{
                            mouseover: () => hoverMarker(i),
                            mouseout: () => hoverMarker(null),
                            click: () => selectMarker(i),
                        }}
                    >
                        <Popup>
                            <div style={{ padding: "12px 16px", minWidth: 180 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                    <div style={{
                                        width: 22, height: 22, borderRadius: "50%",
                                        background: "#e86026", color: "#fff",
                                        fontSize: 11, fontWeight: 700, fontFamily: "system-ui",
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        flexShrink: 0,
                                    }}>
                                        {i + 1}
                                    </div>
                                    <span style={{ fontWeight: 700, fontSize: 13, color: "#fff", lineHeight: 1.3 }}>
                                        {label(s)}
                                    </span>
                                </div>
                                {s.address && (
                                    <div style={{ fontSize: 11, color: "#666", paddingLeft: 30, lineHeight: 1.4 }}>
                                        {[s.address, s.city, s.state].filter(Boolean).join(", ")}
                                    </div>
                                )}
                                {s.excerpt && (
                                    <div style={{
                                        marginTop: 8, paddingLeft: 30,
                                        fontSize: 11, color: "#777", lineHeight: 1.5,
                                        borderTop: "1px solid #1e1e1e", paddingTop: 8,
                                        display: "-webkit-box",
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: "vertical",
                                        overflow: "hidden",
                                    }}>
                                        "{s.excerpt}"
                                    </div>
                                )}
                            </div>
                        </Popup>
                    </Marker>
                ))}

                <MapBounds markers={markers} />
            </MapContainer>
        </div>
    );
}