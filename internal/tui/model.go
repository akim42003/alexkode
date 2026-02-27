package tui

import (
	"v0/internal/config"
	"v0/internal/problem"
	"v0/internal/progress"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

type view int

const (
	viewMainMenu view = iota
	viewProblemList
	viewProblemDetail
	viewTestResults
)

type filterType int

const (
	filterByCategory filterType = iota
	filterByDifficulty
	filterByStatus
	filterAll
)

// Messages
type testResultMsg struct {
	result *problem.TestResult
	err    error
}

type editorFinishedMsg struct {
	err error
}

type Model struct {
	width  int
	height int

	currentView view

	// Data
	problems []problem.Problem
	progress *progress.Progress
	config   *config.Config

	// Main menu
	menuLevel  int      // 0 = filter type, 1 = filter value
	menuCursor int
	menuItems  []string // current items being displayed
	filterType filterType

	// Problem list
	listCursor int
	listOffset int
	filtered   []problem.Problem

	// Problem detail
	viewport     viewport.Model
	vpReady      bool
	showSolution bool
	selectedIdx  int // index into filtered

	// Test results
	testResults *problem.TestResult
	testRunning bool
	testVP      viewport.Model
	testVPReady bool

	// Paths
	problemsDir string
	userCodeDir string
}

func NewModel(problemsDir, userCodeDir, progressPath, configPath string) Model {
	problems, _ := problem.LoadProblems(problemsDir)

	m := Model{
		problems:    problems,
		progress:    progress.Load(progressPath),
		config:      config.Load(configPath),
		currentView: viewMainMenu,
		menuLevel:   0,
		menuCursor:  0,
		problemsDir: problemsDir,
		userCodeDir: userCodeDir,
	}
	m.menuItems = m.topMenuItems()
	return m
}

func (m Model) topMenuItems() []string {
	return []string{"By Category", "By Difficulty", "By Status", "All Problems"}
}

func (m Model) Init() tea.Cmd {
	return nil
}
