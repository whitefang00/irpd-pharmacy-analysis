import { useEffect, useRef, useState } from "react";
import type { ChatEvent, Usage } from "../api";

type Item =
  | { kind: "tool"; id: string; name: string; args: unknown; output?: unknown; truncated?: boolean }
  | { kind: "message"; text: string };

interface Turn {
  question: string;
  items: Item[];
  answer: string | null;
  meta: { model: string; stopped: string; usage: Usage } | null;
  error: { code: string; message: string } | null;
  demo: boolean;
}

// The question of the recorded example (src/irpd/api/demo_transcript.json). A test checks they match.
const DEMO_QUESTION = "Paracetamol 500mg Nova is out. What should customers get instead, and how much Nova should I reorder?";
const SUGGESTIONS = [
  "What should I reorder or move downstairs this week, and why?",
  "Which fast sellers were out of stock longest in the last three months, and what did that cost in sales?",
  "When Salbutamol 4mg Graha Farma runs out, is there another brand customers can switch to?",
  "How much did we sell in August 2025?",
];

/** POST /chat and call onEvent for each server-sent event as it arrives. */
async function stream(body: object, demo: boolean, onEvent: (e: ChatEvent) => void, signal: AbortSignal) {
  const r = await fetch(`/api/chat${demo ? "?demo=1" : ""}`, {
    method: "POST", signal,
    headers: { accept: "text/event-stream", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value.replace(/\r/g, "");
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const event = block.match(/^event: (.*)$/m)?.[1];
      const data = block.match(/^data: (.*)$/m)?.[1];
      if (event && data) onEvent({ event, data: JSON.parse(data) } as ChatEvent);
    }
  }
}

function apply(t: Turn, e: ChatEvent): Turn {
  switch (e.event) {
    case "tool_call":
      return { ...t, items: [...t.items, { kind: "tool", ...e.data }] };
    case "tool_result":
      return { ...t, items: t.items.map((x) => x.kind === "tool" && x.id === e.data.id
        ? { ...x, output: e.data.output, truncated: e.data.truncated } : x) };
    case "message":
      return { ...t, items: [...t.items, { kind: "message", text: e.data.text }] };
    case "answer":
      return { ...t, answer: e.data.text };
    case "done":
      return { ...t, meta: e.data };
    case "error":
      return { ...t, error: e.data };
  }
}

export function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement | null>(null);

  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [turns]);

  async function ask(question: string, demo = false) {
    question = demo ? DEMO_QUESTION : question.trim();
    if (!question || busy) return;
    const history = turns.filter((t) => t.answer && !t.error && !t.demo)
      .map((t) => ({ question: t.question, answer: t.answer }));
    const blank: Turn = { question, items: [], answer: null, meta: null, error: null, demo };
    setTurns((ts) => [...ts, blank]);
    setInput("");
    setBusy(true);
    const update = (f: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), f(ts[ts.length - 1])]);
    abort.current = new AbortController();
    try {
      await stream({ question, history }, demo, (e) => update((t) => apply(t, e)), abort.current.signal);
      update((t) => t.answer || t.meta || t.error ? t
        : { ...t, error: { code: "network", message: "The connection closed before an answer came back." } });
    } catch (err) {
      if (!abort.current.signal.aborted) update((t) => ({ ...t, error: { code: "network", message: String(err) } }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="stack" style={{ gap: 10 }}>
        <h1>Ask the agent</h1>
        <p className="lede">
          The agent answers by calling tools on the data up to 30 June 2025. Every number in its
          answer must come from a tool output. Each call is shown below so you can check it.
        </p>
      </section>

      <div className="stack chat">
        {turns.map((t, i) => <TurnView key={i} t={t} live={busy && i === turns.length - 1} onDemo={() => ask("", true)} />)}
        <div ref={end} />

        <form className="ask" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
          <textarea value={input} rows={2} aria-label="Your question" placeholder="Ask about stock, sales, suppliers or prices…"
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input); } }} />
          <button type="submit" disabled={busy || !input.trim()}>{busy ? "Working…" : "Ask"}</button>
        </form>
        <div className="chips">
          {SUGGESTIONS.map((q) => (
            <button key={q} className="chip" disabled={busy} onClick={() => ask(q)}>{q}</button>
          ))}
          <button className="chip quiet" disabled={busy} onClick={() => ask("", true)}>
            Play the example transcript (no model call)
          </button>
        </div>
      </div>
    </div>
  );
}

function TurnView({ t, live, onDemo }: { t: Turn; live: boolean; onDemo: () => void }) {
  const calls = t.items.filter((x) => x.kind === "tool").length;
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="bubble q">{t.question}</div>
      {t.items.map((x, i) => x.kind === "message"
        ? <p key={i} className="small" style={{ padding: "0 4px" }}>{x.text}</p>
        : <ToolCard key={x.id} item={x} />)}
      {live && !t.answer && !t.error && <p className="small muted" role="status">Working…</p>}
      {t.answer && <div className="bubble a prose"><Prose text={t.answer} /></div>}
      {t.meta && !t.answer && t.meta.stopped === "max_steps" && (
        <p className="error">The agent ran out of steps before answering.</p>)}
      {t.error && (
        <div className="error-box">
          <b>{t.error.code === "budget" ? "Spend cap reached." : t.error.code === "no_key" ? "No API key." : "Couldn't answer."}</b>{" "}
          {t.error.message}
          {t.error.code === "no_key" && <> <button className="linkish" onClick={onDemo}>Play the example instead.</button></>}
        </div>
      )}
      {t.meta && (
        <p className="small muted">
          {t.demo ? `Recorded ${t.meta.model} answer, no model called` : t.meta.model} · {calls} tool call{calls === 1 ? "" : "s"}
          {!t.demo && <> · ${t.meta.usage.cost_usd.toFixed(4)} · {t.meta.usage.input_tokens.toLocaleString()} in / {t.meta.usage.output_tokens.toLocaleString()} out tokens</>}
        </p>
      )}
    </div>
  );
}

function ToolCard({ item }: { item: Extract<Item, { kind: "tool" }> }) {
  const args = item.args && typeof item.args === "object"
    ? Object.entries(item.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")
    : String(item.args ?? "");
  const pending = item.output === undefined;
  return (
    <details className="tool">
      <summary>
        <code>{item.name}</code>
        <span className="small muted args">{args}</span>
        {pending && <span className="small muted running">running…</span>}
      </summary>
      <div className="lbl">Arguments</div>
      <pre>{JSON.stringify(item.args, null, 2)}</pre>
      <div className="lbl">Output{item.truncated ? " (shortened here; the agent saw all of it)" : ""}</div>
      <pre>{pending ? "…" : typeof item.output === "string" ? item.output : JSON.stringify(item.output, null, 2)}</pre>
    </details>
  );
}

/** Just enough markdown for the agent's answers: paragraphs, "- " lists and **bold**. */
function Prose({ text }: { text: string }) {
  const bold = (line: string) => line.split(/\*\*(.+?)\*\*/g).map((part, i) => i % 2 ? <strong key={i}>{part}</strong> : part);
  return <>{text.trim().split(/\n\s*\n/).map((block, i) => {
    const lines = block.split("\n");
    const items = lines.filter((l) => /^\s*[-*] /.test(l));
    const head = lines.filter((l) => !/^\s*[-*] /.test(l));
    return (
      <div key={i}>
        {head.map((l, j) => <p key={j}>{bold(l)}</p>)}
        {items.length > 0 && <ul>{items.map((l, j) => <li key={j}>{bold(l.replace(/^\s*[-*] /, ""))}</li>)}</ul>}
      </div>
    );
  })}</>;
}
