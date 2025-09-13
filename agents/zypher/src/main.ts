import { ZypherAgent, OpenAIModelProvider, runAgentInTerminal } from "jsr:@corespeed/zypher@0.2.1";

const apiKey = Deno.env.get("GROQ_API_KEY");
if (!apiKey) {
  console.error("GROQ_API_KEY missing. Add it to .env or export in shell.");
  Deno.exit(1);
}

// Groq uses OpenAI-compatible API surface
const provider = new OpenAIModelProvider({
  apiKey,
  baseUrl: Deno.env.get("GROQ_BASE_URL") ?? "https://api.groq.com/openai/v1",
});

const agent = new ZypherAgent(
  provider,
  {
    name: "LMS Zypher Agent",
    description: "Minimal agent wired to Groq via OpenAI-compatible API",
    systemPrompt: "You are a helpful LMS assistant.",
  },
  {} as any,
);

const MODEL = Deno.env.get("GROQ_MODEL") ?? "llama-3.1-70b-versatile";

await runAgentInTerminal(agent, MODEL);
