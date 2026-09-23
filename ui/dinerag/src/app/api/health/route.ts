import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
    const url = req.nextUrl.searchParams.get("url");
    if (!url) return NextResponse.json({ ok: false }, { status: 400 });

    try {
        const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(8000) });
        return NextResponse.json({ ok: res.ok });
    } catch {
        return NextResponse.json({ ok: false });
    }
}