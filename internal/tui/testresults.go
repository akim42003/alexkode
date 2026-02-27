package tui

func (m Model) renderTestResults() string {
	if m.selectedIdx >= len(m.filtered) {
		return "No problem selected."
	}

	p := m.filtered[m.selectedIdx]
	s := m.renderHeader(p.Category, p.Title, "Test Results")

	if !m.testVPReady {
		return s + "\nLoading..."
	}

	s += m.testVP.View() + "\n"
	s += "\n" + styleHelp.Render("  [↑/↓] Scroll  [Esc] Back  [q] Quit")

	return s
}
