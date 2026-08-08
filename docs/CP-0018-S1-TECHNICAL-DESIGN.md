# CP-0018 Sprint 1 Technical Design

## Goal

Create the permanent product navigation shell.

## Components

ProductAppWindow
- owns central application container
- exposes product routes
- renders active screen

NavigationController
- route registry
- current route
- navigation history
- route validation

ProductSession
- app-level session state
- selected camera
- calibration profile
- developer mode
- active match id

Theme
- shared stylesheet
- product typography / spacing baseline

## Routes

HOME
START_MATCH
HISTORY
SETTINGS
DIAGNOSTICS

Sprint 2 will replace START_MATCH placeholder with the Match Wizard.
