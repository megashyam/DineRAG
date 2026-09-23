"use client";

import { createContext, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark";
type Ctx = { theme: Theme; toggle: () => void };

const ThemeContext = createContext<Ctx>({ theme: "light", toggle: () => { } });

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const [theme, setTheme] = useState<Theme>("light");

    useEffect(() => {
        const saved = (localStorage.getItem("fg-theme") as Theme) ?? "light";
        apply(saved);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    function apply(t: Theme) {
        setTheme(t);
        document.documentElement.classList.toggle("dark", t === "dark");
        localStorage.setItem("fg-theme", t);
    }

    return (
        <ThemeContext.Provider value={{ theme, toggle: () => apply(theme === "light" ? "dark" : "light") }}>
            {children}
        </ThemeContext.Provider>
    );
}

export const useTheme = () => useContext(ThemeContext);