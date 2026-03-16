**UI Components **
1.AppShell
Root container for layout , shared state, routing and session persistence
Responsibilities :
mount global layout
provide top-level state
coordinate panel visibility
route between learn, simulate and compare modes
Inputs :
appTitle , theme , activeWorkspace , sidebarCollapsed,bottomPanelVisible,activeDiagramId
Outputs:
app-wide layout state , diagram selection state , session snapshot
