package tui

func (m Model) renderProblemDetail() string {
	if m.selectedIdx >= len(m.filtered) {
		return "No problem selected."
	}

	p := m.filtered[m.selectedIdx]

	// Build breadcrumb
	viewLabel := p.Title
	if m.showSolution {
		viewLabel = p.Title + " (Solution)"
	}
	s := m.renderHeader(p.Category, viewLabel)

	if !m.vpReady {
		return s + "\nLoading..."
	}

	s += m.viewport.View() + "\n"

	// Status line
	status := ""
	if m.progress.IsCompleted(p.ID) {
		status = styleCompleted.Render("[completed]") + "  "
	}

	if m.testRunning {
		status += styleStats.Render("Running tests...")
	}

	s += status

	// Help bar
	helpItems := "[e] Edit  [r] Run Tests"
	if m.showSolution {
		helpItems += "  [s] Hide Solution"
	} else {
		helpItems += "  [s] Show Solution"
	}
	helpItems += "  [c] Toggle Complete  [Esc] Back"
	s += "\n" + styleHelp.Render("  "+helpItems)

	return s
}
