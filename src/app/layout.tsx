import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI Trader — T-Bank Sandbox",
  description:
    "Дашборд автономного AI-трейдера на RL-алгоритмах. Торгует на демо-счёте T-Банка 24/7.",
  keywords: [
    "AI trader",
    "reinforcement learning",
    "T-Bank",
    "T-Invest",
    "trading bot",
    "sandbox",
  ],
  authors: [{ name: "AI Trader" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
