import { useEffect, useState } from "react";

/** Load once per key; keeps the previous data while a new key loads. */
export function useLoad<T>(load: () => Promise<T>, key: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    load().then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e) => live && setError(String(e)))
      .finally(() => live && setLoading(false));
    return () => { live = false; };
  }, [key]);
  return { data, error, loading };
}
