import { useEffect, useState } from "react";
import { ChatPage } from "./pages/Chat";
import { DrugPageView } from "./pages/Drug";
import { OverviewPage } from "./pages/Overview";
import { ReplayPage } from "./pages/Replay";

// Hash routes: #/  #/drug/CODE  #/replay  #/chat
function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [r, setR] = useState(read);
  useEffect(() => {
    const on = () => { setR(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}

function ThemeToggle() {
  const [theme, setTheme] = useState<string | null>(() => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem("theme", theme) : localStorage.removeItem("theme"); } catch { /* ignore */ }
  }, [theme]);
  const dark = theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <button className="theme" onClick={() => setTheme(dark ? "light" : "dark")}
            aria-label={`Switch to ${dark ? "light" : "dark"} mode`}>
      {dark ? "Light" : "Dark"}
    </button>
  );
}

export function App() {
  const [page, arg] = useRoute();
  const link = (href: string, label: string, on: boolean) => (
    <a className={`link${on ? " active" : ""}`} href={href} aria-current={on ? "page" : undefined}>{label}</a>
  );
  return (
    <div className="shell">
      <nav className="nav">
        <a className="brand" href="#/">Pharmacy analyst</a>
        {link("#/", "Overview", !page)}
        {link("#/drug", "Drugs", page === "drug")}
        {link("#/replay", "Agent vs pharmacist", page === "replay")}
        {link("#/chat", "Chat", page === "chat")}
        <ThemeToggle />
      </nav>
      {page === "drug" ? <DrugPageView code={arg ? decodeURIComponent(arg) : undefined} />
        : page === "replay" ? <ReplayPage />
        : page === "chat" ? <ChatPage />
        : <OverviewPage />}
    </div>
  );
}
