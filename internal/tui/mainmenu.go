package tui

import (
	"fmt"

	"v0/internal/problem"
)

func (m Model) renderMainMenu() string {
	var s string

	if m.menuLevel == 0 {
		s = m.renderHeader()
	} else {
		label := m.topMenuItems()[int(m.filterType)]
		s = m.renderHeader(label)
	}

	s += "\n"

	for i, item := range m.menuItems {
		cursor := "  "
		if m.menuCursor == i {
			cursor = styleCursor.Render("> ")
		}

		line := item

		// Add counts for filter values at level 1
		if m.menuLevel == 1 {
			switch m.filterType {
			case filterByCategory:
				cat := problem.FilterByCategory(m.problems, item)
				completed := 0
				for _, p := range cat {
					if m.progress.IsCompleted(p.ID) {
						completed++
					}
				}
				count := fmt.Sprintf("  %d/%d", completed, len(cat))
				if m.menuCursor == i {
					line = styleSelected.Render(item) + styleMuted(count)
				} else {
					line = styleNormal.Render(item) + styleMuted(count)
				}
			case filterByDifficulty:
				diff := problem.FilterByDifficulty(m.problems, item)
				completed := 0
				for _, p := range diff {
					if m.progress.IsCompleted(p.ID) {
						completed++
					}
				}
				count := fmt.Sprintf("  %d/%d", completed, len(diff))
				badge := difficultyStyle(item).Render(item)
				if m.menuCursor == i {
					line = badge + styleMuted(count)
				} else {
					line = badge + styleMuted(count)
				}
			case filterByStatus:
				if item == "Completed" {
					n := m.progress.CompletedCount()
					line = styleCompleted.Render(fmt.Sprintf("%s  (%d)", item, n))
				} else {
					n := len(m.problems) - m.progress.CompletedCount()
					line = styleIncomplete.Render(fmt.Sprintf("%s  (%d)", item, n))
				}
			}
		} else {
			if m.menuCursor == i {
				line = styleSelected.Render(item)
			} else {
				line = styleNormal.Render(item)
			}
		}

		s += cursor + line + "\n"
	}

	s += "\n"

	if m.menuLevel == 0 {
		// Progress bar
		completed := m.progress.CompletedCount()
		total := len(m.problems)
		barWidth := 30
		filled := 0
		if total > 0 {
			filled = (completed * barWidth) / total
		}
		bar := styleCompleted.Render(repeat("█", filled)) +
			styleIncomplete.Render(repeat("░", barWidth-filled))
		s += fmt.Sprintf("  Progress: %s %d/%d\n", bar, completed, total)
	}

	s += "\n"

	if m.menuLevel == 0 {
		s += styleHelp.Render("  [↑/↓] Navigate  [Enter] Select  [q] Quit")
	} else {
		s += styleHelp.Render("  [↑/↓] Navigate  [Enter] Select  [Esc] Back  [q] Quit")
	}

	return s
}

func styleMuted(s string) string {
	return styleIncomplete.Render(s)
}

func repeat(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
