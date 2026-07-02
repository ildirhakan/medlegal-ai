import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// PII anonymisation (server-side)
function anonymisePII(t: string): string {
  if (!t) return t;
  return t
    .replace(/\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b/g, "[NHS-REDACTED]")
    .replace(/\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b/gi, "[POSTCODE-REDACTED]")
    .replace(/\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b/g, "[DOB-REDACTED]")
    .replace(/\b[\w.-]+@[\w.-]+\.\w{2,}\b/g, "[EMAIL-REDACTED]")
    .replace(/\b(?:0|\+44)\s*\d[\d\s]{8,12}\b/g, "[PHONE-REDACTED]");
}

function buildPrompt(ct: string, cs: string): string {
  return `You are a UK clinical negligence legal analyst. Analyse this case under UK law.

CASE TYPE: ${ct}
CASE SUMMARY: ${anonymisePII(cs)}

Apply these legal tests:
1. DUTY OF CARE: Confirmed / Arguable / Not Established
2. BREACH (BOLAM TEST): Clear Breach / Arguable Breach / No Breach
3. BOLITHO DEFENSIBILITY: Not Defensible / Partially Defensible / Fully Defensible
4. CAUSATION: Established / Arguable / Not Established
5. OVERALL RISK SCORE: 0-100
6. RECOMMENDATION: "Proceed to Expert Report" / "Borderline - Consider Further" / "Do Not Pursue"
7. DETAILED REASONING: 4-6 sentence analysis.

Respond ONLY in JSON:
{"duty_of_care":"string","breach_bolam":"string","bolitho_defensibility":"string","causation":"string","risk_score":0,"recommendation":"string","reasoning":"string"}`;
}

async function callGPT4(prompt: string) {
  try {
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${Deno.env.get("OPENAI_API_KEY")}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3,
        max_tokens: 1200,
      }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error.message);
    return JSON.parse(d.choices[0].message.content.replace(/```json|```/g, "").trim());
  } catch (e) {
    console.error("GPT4:", e.message);
    return null;
  }
}

async function callGemini(prompt: string) {
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${Deno.env.get("GEMINI_API_KEY")}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.3, maxOutputTokens: 1200 },
        }),
      }
    );
    const d = await r.json();
    if (d.error) throw new Error(d.error.message);
    return JSON.parse(d.candidates[0].content.parts[0].text.replace(/```json|```/g, "").trim());
  } catch (e) {
    console.error("Gemini:", e.message);
    return null;
  }
}

async function callDeepSeek(prompt: string) {
  try {
    const r = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${Deno.env.get("DEEPSEEK_API_KEY")}`,
      },
      body: JSON.stringify({
        model: "deepseek-chat",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3,
        max_tokens: 1200,
      }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error.message);
    return JSON.parse(d.choices[0].message.content.replace(/```json|```/g, "").trim());
  } catch (e) {
    console.error("DeepSeek:", e.message);
    return null;
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { caseType, caseSummary } = await req.json();

    if (!caseType || !caseSummary) {
      return new Response(
        JSON.stringify({ error: "Missing caseType or caseSummary" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const prompt = buildPrompt(caseType, caseSummary);

    // Run models in parallel
    const [gpt4, gemini, deepseek] = await Promise.all([
      callGPT4(prompt),
      callGemini(prompt),
      callDeepSeek(prompt),
    ]);

    const results: Record<string, any> = {};
    if (gpt4) results.gpt4 = gpt4;
    if (gemini) results.gemini = gemini;
    if (deepseek) results.deepseek = deepseek;

    // Compute average + consistency
    const valid = Object.values(results);
    let avg = 0, consistency = 0;
    if (valid.length > 0) {
      const scores = valid.map((v: any) => v.risk_score);
      avg = Math.round(scores.reduce((a: number, b: number) => a + b, 0) / scores.length);
      if (scores.length > 1) {
        const mean = scores.reduce((a: number, b: number) => a + b, 0) / scores.length;
        const variance = scores.reduce((a: number, s: number) => a + Math.pow(s - mean, 2), 0) / scores.length;
        consistency = Math.round(Math.max(0, 100 - Math.sqrt(variance) * 2));
      } else {
        consistency = 75;
      }
    }

    return new Response(
      JSON.stringify({ results, avgScore: avg, consistency }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e.message }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});