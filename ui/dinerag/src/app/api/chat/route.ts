import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const city = req.headers.get("x-vercel-ip-city") || "unknown";
    const country = req.headers.get("x-vercel-ip-country") || "unknown";

    // Ensure this matches your Python Generator's port (defaulted to 8001 in my previous snippet)
    // If you are running locally, it might be http://localhost:8001/generate
    const PYTHON_API_URL =
      // process.env.GENERATOR_URL || "http://localhost:9000/generate";
      process.env.GENERATOR_URL || "https://megumind6172--food-rag-generator-serve.modal.run";

    const response = await fetch(PYTHON_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
      body: JSON.stringify({
        query: body.query,
        city: body.city || null,   // Explicitly handle nulls
        state: body.state || null, // Pass state if available
        restaurant: body.restaurant || null,
        address: body.address || null,
        top_k: body.top_k || 12,
      }),
    });

    if (!response.ok || !response.body) {
      const errorText = await response.text();
      console.error("Python Backend Error:", errorText);
      return NextResponse.json(
        { error: "Python backend error", details: errorText },
        { status: response.status || 500 }
      );
    }

    // Pass the stream directly back to the client
    return new NextResponse(response.body, {
      status: 200,
      headers: {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (err) {
    console.error("API Route Error:", err);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  return NextResponse.json({
    city: req.headers.get("x-vercel-ip-city"),
    region: req.headers.get("x-vercel-ip-country-region"),
    country: req.headers.get("x-vercel-ip-country"),
    ip: req.headers.get("x-forwarded-for"),
  });
}