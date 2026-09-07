import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { toolDefinitions, toolImplementations } from './tools.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-5';
const PORT = process.env.PORT || 3000;
const MAX_TOKENS = 1536;
const MAX_TOOL_ROUNDS = 5;

const SYSTEM_PROMPT = `You are a clinical reference assistant for licensed healthcare professionals.

Scope and limits (do not deviate from these):
- You provide reference information only: drug labeling, medical literature, and drug nomenclature, sourced live via your tools (FDA, PubMed, RxNorm).
- You are NOT a diagnostic tool. Never diagnose a specific patient, and never recommend a specific treatment or dose for a specific patient case.
- For any drug-safety, dosing, or interaction question, prefer calling a tool over answering from memory, and cite what the tool returned.
- If a tool returns no data or an error, say so plainly. Do not fill the gap with unverified claims.
- Always state that outputs must be independently verified against current prescribing information and clinical judgment before use.
- Keep answers concise and scannable for a clinician skimming between patients.`;

if (!API_KEY) {
  console.error('ANTHROPIC_API_KEY is not set. Set it in this terminal session before starting the server.');
  process.exit(1);
}

async function callClaude(messages) {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      system: SYSTEM_PROMPT,
      tools: toolDefinitions,
      messages
    })
  });

  if (!res.ok) {
    const details = await res.text().catch(() => '');
    throw new Error(`Claude API request failed (${res.status}): ${details}`);
  }

  return res.json();
}

/** Runs the tool-use loop for one user turn and returns the final assistant text plus tools used. */
async function runTurn(history) {
  const messages = [...history];
  const toolsUsed = [];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const response = await callClaude(messages);

    if (response.stop_reason !== 'tool_use') {
      const text = response.content
        .filter((block) => block.type === 'text')
        .map((block) => block.text)
        .join('\n');
      return { text, toolsUsed };
    }

    messages.push({ role: 'assistant', content: response.content });

    const toolResults = [];
    for (const block of response.content) {
      if (block.type !== 'tool_use') continue;
      toolsUsed.push({ name: block.name, input: block.input });

      let resultContent;
      try {
        const impl = toolImplementations[block.name];
        if (!impl) throw new Error(`Unknown tool: ${block.name}`);
        const result = await impl(block.input ?? {});
        resultContent = JSON.stringify(result);
      } catch (err) {
        resultContent = JSON.stringify({ error: err.message });
      }

      toolResults.push({
        type: 'tool_result',
        tool_use_id: block.id,
        content: resultContent
      });
    }

    messages.push({ role: 'user', content: toolResults });
  }

  return { text: 'Reached the tool-call limit for this turn without a final answer. Try rephrasing your question.', toolsUsed };
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/chat', async (req, res) => {
  const { history } = req.body;
  if (!Array.isArray(history) || history.length === 0) {
    return res.status(400).json({ error: 'Request body must include a non-empty "history" array.' });
  }

  try {
    const { text, toolsUsed } = await runTurn(history);
    res.json({ reply: text, toolsUsed });
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: err.message });
  }
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Clinical assistant running at http://127.0.0.1:${PORT}`);
});
