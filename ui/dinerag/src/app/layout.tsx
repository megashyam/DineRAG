import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";
import ThemeToggle from "@/components/ThemeToggle";
import { ChefHat, BookOpen, MessageSquare } from "lucide-react";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DineRAG",
  description: "RAG powered restaurant discovery",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <style>{`
          .nav-link { transition: all 0.15s; }
          .nav-link:hover { background-color: var(--bg-3); color: var(--text); }
          .nav-link-readme { display: flex; align-items: center; gap: 6px; }
        `}</style>
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`} style={{ margin: 0 }}>
        <ThemeProvider>
          <div style={{ height: "100dvh", display: "flex", flexDirection: "column", background: "var(--bg)", overflow: "hidden" }}>

            {/* ── Global nav ─────────────────────────────────────────── */}
            <header style={{
              flexShrink: 0,
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "0 24px",
              height: 60,
              background: "var(--bg-2)",
              borderBottom: "1px solid var(--border)",
              zIndex: 30,
            }}>

              {/* Brand */}
              <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none", marginRight: "auto" }}>
                <div style={{
                  width: 36, height: 36,
                  background: "#e86026",
                  borderRadius: 10,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "0 0 14px rgba(232,96,38,0.3)",
                  flexShrink: 0,
                }}>
                  <ChefHat className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)", lineHeight: 1 }}>
                    DineRAG
                  </p>
                  <p style={{ margin: 0, fontSize: 10, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 2 }}>
                    RAG-powered restaurant discovery
                  </p>
                </div>
              </Link>

              {/* Nav links */}
              <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Link
                  href="/"
                  className="nav-link"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "6px 12px",
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 500,
                    color: "var(--text-2)",
                    textDecoration: "none",
                  }}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Chat
                </Link>
                <Link
                  href="/readme"
                  className="nav-link"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "6px 12px",
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 500,
                    color: "var(--text-2)",
                    textDecoration: "none",
                  }}
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  README
                </Link>
              </nav>

              <ThemeToggle />
            </header>

            {/* ── Page content ───────────────────────────────────────── */}
            <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
              {children}
            </div>

          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}