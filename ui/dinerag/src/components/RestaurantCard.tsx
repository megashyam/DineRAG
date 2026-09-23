"use client";

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

function getName(s: Source) {
    return (s.name || s.restaurant || "").trim();
}

function getLocation(s: Source) {
    return [s.city, s.state].filter(Boolean).join(", ");
}

export function RestaurantCard({ source, index }: { source: Source; index: number }) {
    const name = getName(source);
    const location = getLocation(source);
    const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${name} ${source.address || ""} ${location}`)}`;
    const yelpUrl = `https://www.yelp.com/search?find_desc=${encodeURIComponent(name)}&find_loc=${encodeURIComponent(location)}`;

    return (
        <div className="restaurant-card" style={{ animationDelay: `${index * 80}ms` }}>
            <style>{`
        .restaurant-card {
          flex-shrink: 0;
          width: 220px;
          background: var(--bg-2);
          border: 1px solid var(--border);
          border-radius: 14px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
          transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
          animation: cardIn 0.4s ease both;
          cursor: default;
        }
        .restaurant-card:hover {
          border-color: rgba(232,96,38,0.5);
          box-shadow: 0 0 0 1px rgba(232,96,38,0.15), 0 8px 24px rgba(0,0,0,0.15);
          transform: translateY(-2px);
        }
        @keyframes cardIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .card-header {
          padding: 12px 14px 10px;
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: flex-start;
          gap: 10px;
        }
        .card-badge {
          width: 26px;
          height: 26px;
          border-radius: 50%;
          background: #e86026;
          color: #fff;
          font-size: 12px;
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          margin-top: 1px;
          font-family: system-ui, sans-serif;
          box-shadow: 0 0 10px rgba(232,96,38,0.3);
        }
        .card-name {
          font-size: 13px;
          font-weight: 700;
          color: var(--text);
          line-height: 1.3;
          font-family: Georgia, 'Times New Roman', serif;
        }
        .card-location {
          font-size: 10px;
          color: var(--text-3);
          margin-top: 3px;
          display: flex;
          align-items: center;
          gap: 4px;
          font-family: system-ui, sans-serif;
        }
        .card-footer {
          display: flex;
          gap: 6px;
          padding: 10px 14px 12px;
        }
        .card-link {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 4px;
          padding: 6px 8px;
          border-radius: 7px;
          font-size: 10px;
          font-weight: 600;
          text-decoration: none;
          transition: background 0.15s, color 0.15s, border-color 0.15s;
          font-family: system-ui, sans-serif;
          background: var(--bg-3);
          border: 1px solid var(--border);
          color: var(--text-2);
        }
        .card-link:hover {
          background: rgba(232,96,38,0.1);
          border-color: #e86026;
          color: #e86026;
        }
        .card-link.yelp:hover {
          background: rgba(196,18,0,0.1);
          border-color: #c41200;
          color: #ff3300;
        }
      `}</style>

            <div className="card-header">
                <div className="card-badge">{index + 1}</div>
                <div style={{ flex: 1, overflow: "hidden" }}>
                    <div className="card-name">{name || "Unknown Restaurant"}</div>
                    {(location || source.address) && (
                        <div className="card-location">
                            <span>📍</span>
                            <span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                {source.address ? `${source.address}${location ? `, ${location}` : ""}` : location}
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {source.excerpt && (
                <div style={{ padding: "10px 14px 6px", flex: 1 }}>
                    <div style={{
                        borderLeft: "2px solid var(--border-2)",
                        paddingLeft: 10,
                        fontSize: "10.5px",
                        color: "var(--text-3)",
                        lineHeight: 1.6,
                        fontStyle: "italic",
                        display: "-webkit-box",
                        WebkitLineClamp: 4,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                    }}>
                        &ldquo;{source.excerpt}&rdquo;
                    </div>
                </div>
            )}

            <div className="card-footer">
                <a href={gmapsUrl} target="_blank" rel="noopener noreferrer"
                    className="card-link" onClick={(e) => e.stopPropagation()}>
                    🗺 Maps
                </a>
                <a href={yelpUrl} target="_blank" rel="noopener noreferrer"
                    className="card-link yelp" onClick={(e) => e.stopPropagation()}>
                    ⭐ Yelp
                </a>
            </div>
        </div>
    );
}

export function RestaurantCardRow({ sources }: { sources: Source[] }) {
    const valid = sources.filter((s) => getName(s).length > 0);
    const seen = new Set<string>();
    const unique = sources.filter((s) => {
        const name = getName(s).toLowerCase();

        if (!name || seen.has(name)) return false;
        seen.add(name);
        return true;
    });
    if (unique.length === 0) return null;
    if (valid.length === 0) return null;

    return (
        <div style={{
            display: "flex",
            gap: 10,
            overflowX: "auto",
            paddingBottom: 6,
            marginBottom: 12,
            scrollbarWidth: "thin",
            scrollbarColor: "var(--border) transparent",
        }}>
            {valid.map((s, i) => (
                <RestaurantCard key={i} source={s} index={i} />
            ))}
        </div>
    );
}