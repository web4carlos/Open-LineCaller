# CP-0018 Sprint 6 Technical Design — Match Summary & Reporting

## Goal

Complete the first end-to-end product flow:

Home
-> Match Wizard
-> Live
-> Replay
-> End Match
-> Match Summary
-> Save / Export
-> Home

## Components

MatchSummary
- canonical immutable match result

MatchSummaryBuilder
- builds summary from runtime metrics

MatchHistoryStore
- persists summaries as JSONL

MatchReportExporter
- exports one summary as formatted JSON

MatchSummaryScreen
- product UI for reviewing a completed match

## Safety

Summary/reporting code does not alter officiating decisions.
It only records and presents already-produced evidence.
