package tui

import "github.com/charmbracelet/lipgloss"

var (
	colorPrimary   = lipgloss.Color("#7C3AED") // purple
	colorSecondary = lipgloss.Color("#6366F1") // indigo
	colorSuccess   = lipgloss.Color("#10B981") // green
	colorDanger    = lipgloss.Color("#EF4444") // red
	colorWarning   = lipgloss.Color("#F59E0B") // amber
	colorMuted     = lipgloss.Color("#6B7280") // gray
	colorText      = lipgloss.Color("#E5E7EB") // light gray
	colorBg        = lipgloss.Color("#1F2937") // dark bg

	styleTitle = lipgloss.NewStyle().
			Bold(true).
			Foreground(colorPrimary).
			MarginBottom(1)

	styleBreadcrumb = lipgloss.NewStyle().
			Foreground(colorMuted)

	styleSelected = lipgloss.NewStyle().
			Foreground(colorPrimary).
			Bold(true)

	styleNormal = lipgloss.NewStyle().
			Foreground(colorText)

	styleCursor = lipgloss.NewStyle().
			Foreground(colorPrimary).
			Bold(true)

	styleCompleted = lipgloss.NewStyle().
			Foreground(colorSuccess)

	styleIncomplete = lipgloss.NewStyle().
			Foreground(colorMuted)

	styleDiffEasy = lipgloss.NewStyle().
			Foreground(colorSuccess).
			Bold(true)

	styleDiffMedium = lipgloss.NewStyle().
			Foreground(colorWarning).
			Bold(true)

	styleDiffHard = lipgloss.NewStyle().
			Foreground(colorDanger).
			Bold(true)

	styleHelp = lipgloss.NewStyle().
			Foreground(colorMuted).
			MarginTop(1)

	styleError = lipgloss.NewStyle().
			Foreground(colorDanger)

	stylePass = lipgloss.NewStyle().
			Foreground(colorSuccess).
			Bold(true)

	styleFail = lipgloss.NewStyle().
			Foreground(colorDanger).
			Bold(true)

	styleHeader = lipgloss.NewStyle().
			Bold(true).
			Foreground(colorPrimary).
			BorderStyle(lipgloss.NormalBorder()).
			BorderBottom(true).
			BorderForeground(colorMuted).
			Width(60)

	styleStats = lipgloss.NewStyle().
			Foreground(colorSecondary)
)

func difficultyStyle(diff string) lipgloss.Style {
	switch diff {
	case "Easy":
		return styleDiffEasy
	case "Medium":
		return styleDiffMedium
	case "Hard":
		return styleDiffHard
	default:
		return styleNormal
	}
}
