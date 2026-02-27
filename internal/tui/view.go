package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) View() string {
	switch m.currentView {
	case viewMainMenu:
		return m.renderMainMenu()
	case viewProblemList:
		return m.renderProblemList()
	case viewProblemDetail:
		return m.renderProblemDetail()
	case viewTestResults:
		return m.renderTestResults()
	default:
		return "Unknown view"
	}
}

func (m Model) renderHeader(breadcrumbs ...string) string {
	title := "Deep-ML 60"
	path := title
	if len(breadcrumbs) > 0 {
		path = title + styleBreadcrumb.Render(" > "+strings.Join(breadcrumbs, " > "))
	}

	completed := m.progress.CompletedCount()
	stats := styleStats.Render(fmt.Sprintf("%d/60 completed", completed))

	// Right-align stats
	padding := m.width - lipgloss.Width(path) - lipgloss.Width(stats) - 2
	if padding < 2 {
		padding = 2
	}

	header := styleTitle.Render(path) + strings.Repeat(" ", padding) + stats
	divider := styleBreadcrumb.Render(strings.Repeat("─", min(m.width, 70)))

	return header + "\n" + divider + "\n"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
