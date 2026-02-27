package tui

import (
	"fmt"
)

func (m Model) renderProblemList() string {
	// Build breadcrumb
	var crumb string
	switch m.filterType {
	case filterByCategory:
		if len(m.filtered) > 0 {
			crumb = m.filtered[0].Category
		}
	case filterByDifficulty:
		if len(m.filtered) > 0 {
			crumb = m.filtered[0].Difficulty
		}
	case filterByStatus:
		crumb = "Filtered"
	case filterAll:
		crumb = "All Problems"
	}

	s := m.renderHeader(crumb)
	s += "\n"

	if len(m.filtered) == 0 {
		s += styleIncomplete.Render("  No problems found.\n")
		s += "\n"
		s += styleHelp.Render("  [Esc] Back  [q] Quit")
		return s
	}

	maxVisible := m.height - 8
	if maxVisible < 1 {
		maxVisible = 5
	}

	end := m.listOffset + maxVisible
	if end > len(m.filtered) {
		end = len(m.filtered)
	}

	for i := m.listOffset; i < end; i++ {
		p := m.filtered[i]

		cursor := "  "
		if i == m.listCursor {
			cursor = styleCursor.Render("> ")
		}

		// Completion indicator
		status := styleIncomplete.Render("  ")
		if m.progress.IsCompleted(p.ID) {
			status = styleCompleted.Render("+ ")
		}

		// Problem number and title
		numStr := fmt.Sprintf("%2d. ", p.Number)
		title := p.Title

		// Pad title for alignment
		titleWidth := 38
		if len(title) > titleWidth {
			title = title[:titleWidth-1] + "~"
		}
		for len(title) < titleWidth {
			title += " "
		}

		// Difficulty badge
		diff := difficultyStyle(p.Difficulty).Render(p.Difficulty)

		if i == m.listCursor {
			line := cursor + status +
				styleSelected.Render(numStr) +
				styleSelected.Render(title) +
				" " + diff
			s += line + "\n"
		} else {
			line := cursor + status +
				styleNormal.Render(numStr) +
				styleNormal.Render(title) +
				" " + diff
			s += line + "\n"
		}
	}

	// Scroll indicator
	if len(m.filtered) > maxVisible {
		s += fmt.Sprintf("\n  %d-%d of %d",
			m.listOffset+1,
			end,
			len(m.filtered))
	}

	s += "\n\n"
	s += styleHelp.Render("  [↑/↓] Navigate  [Enter] Open  [Esc] Back  [q] Quit")

	return s
}
