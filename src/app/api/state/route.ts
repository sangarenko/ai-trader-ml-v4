import { NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';

export const dynamic = 'force-static';

// Preview-only endpoint: returns a captured snapshot of the real
// /api/state payload from the AI Trader sandbox. When this dashboard is
// deployed on the trader server, replace this route with the real state
// producer (or point the client at the existing /api/state).
export async function GET() {
  const file = path.join(process.cwd(), 'data', 'state-snapshot.json');
  try {
    const raw = fs.readFileSync(file, 'utf-8');
    const data = JSON.parse(raw);
    return NextResponse.json(data, {
      headers: { 'cache-control': 'no-store' },
    });
  } catch {
    return NextResponse.json(
      { error: 'snapshot not found', bots: [], botLiveStats: {}, agents: [], logs: [] },
      { status: 500 },
    );
  }
}
