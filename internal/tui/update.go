package tui

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"v0/internal/problem"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		if m.currentView == viewProblemDetail && m.vpReady {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height - 6
		}
		if m.currentView == viewTestResults && m.testVPReady {
			m.testVP.Width = msg.Width
			m.testVP.Height = msg.Height - 6
		}
		return m, nil

	case testResultMsg:
		m.testRunning = false
		if msg.err != nil {
			m.testResults = &problem.TestResult{
				Error: msg.err.Error(),
			}
		} else {
			m.testResults = msg.result

			// Auto-mark completed if all tests pass
			if msg.result != nil && msg.result.Passed {
				p := m.filtered[m.selectedIdx]
				m.progress.MarkCompleted(p.ID)
				m.progress.Save()
			}
		}
		m.currentView = viewTestResults
		m.initTestResultsViewport()
		return m, nil

	case editorFinishedMsg:
		return m, nil

	case tea.KeyMsg:
		// Global quit
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}

		switch m.currentView {
		case viewMainMenu:
			return m.updateMainMenu(msg)
		case viewProblemList:
			return m.updateProblemList(msg)
		case viewProblemDetail:
			return m.updateProblemDetail(msg)
		case viewTestResults:
			return m.updateTestResults(msg)
		}
	}

	return m, nil
}

// --- Main Menu Update ---

func (m Model) updateMainMenu(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q":
		if m.menuLevel == 0 {
			return m, tea.Quit
		}
		m.menuLevel = 0
		m.menuCursor = 0
		m.menuItems = m.topMenuItems()

	case "esc":
		if m.menuLevel == 1 {
			m.menuLevel = 0
			m.menuCursor = 0
			m.menuItems = m.topMenuItems()
		}

	case "up", "k":
		if m.menuCursor > 0 {
			m.menuCursor--
		}

	case "down", "j":
		if m.menuCursor < len(m.menuItems)-1 {
			m.menuCursor++
		}

	case "enter":
		if m.menuLevel == 0 {
			switch m.menuCursor {
			case 0: // By Category
				m.filterType = filterByCategory
				m.menuItems = problem.Categories(m.problems)
				m.menuLevel = 1
				m.menuCursor = 0
			case 1: // By Difficulty
				m.filterType = filterByDifficulty
				m.menuItems = []string{"Easy", "Medium", "Hard"}
				m.menuLevel = 1
				m.menuCursor = 0
			case 2: // By Status
				m.filterType = filterByStatus
				m.menuItems = []string{"Completed", "Incomplete"}
				m.menuLevel = 1
				m.menuCursor = 0
			case 3: // All Problems
				m.filterType = filterAll
				m.filtered = m.problems
				m.currentView = viewProblemList
				m.listCursor = 0
				m.listOffset = 0
			}
		} else {
			// Level 1: apply filter and go to problem list
			value := m.menuItems[m.menuCursor]
			switch m.filterType {
			case filterByCategory:
				m.filtered = problem.FilterByCategory(m.problems, value)
			case filterByDifficulty:
				m.filtered = problem.FilterByDifficulty(m.problems, value)
			case filterByStatus:
				if value == "Completed" {
					m.filtered = problem.FilterByCompleted(m.problems, m.progress.Completed, true)
				} else {
					m.filtered = problem.FilterByCompleted(m.problems, m.progress.Completed, false)
				}
			}
			m.currentView = viewProblemList
			m.listCursor = 0
			m.listOffset = 0
		}
	}

	return m, nil
}

// --- Problem List Update ---

func (m Model) updateProblemList(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	maxVisible := m.height - 6
	if maxVisible < 1 {
		maxVisible = 1
	}

	switch msg.String() {
	case "q":
		return m, tea.Quit

	case "esc":
		m.currentView = viewMainMenu
		m.menuLevel = 0
		m.menuCursor = 0
		m.menuItems = m.topMenuItems()

	case "up", "k":
		if m.listCursor > 0 {
			m.listCursor--
			if m.listCursor < m.listOffset {
				m.listOffset = m.listCursor
			}
		}

	case "down", "j":
		if m.listCursor < len(m.filtered)-1 {
			m.listCursor++
			if m.listCursor >= m.listOffset+maxVisible {
				m.listOffset = m.listCursor - maxVisible + 1
			}
		}

	case "enter":
		if len(m.filtered) > 0 {
			m.selectedIdx = m.listCursor
			m.showSolution = false
			m.currentView = viewProblemDetail
			m.initDetailViewport()
			m.progress.SetLastOpened(m.filtered[m.selectedIdx].ID)
			m.progress.Save()
		}
	}

	return m, nil
}

// --- Problem Detail Update ---

func (m Model) updateProblemDetail(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q":
		return m, tea.Quit

	case "esc":
		m.currentView = viewProblemList

	case "s":
		m.showSolution = !m.showSolution
		m.initDetailViewport()

	case "e":
		p := m.filtered[m.selectedIdx]
		return m, m.openEditor(p)

	case "r":
		if !m.testRunning {
			m.testRunning = true
			p := m.filtered[m.selectedIdx]
			return m, m.runTestsCmd(p)
		}

	case "c":
		// Toggle completion
		p := m.filtered[m.selectedIdx]
		if m.progress.IsCompleted(p.ID) {
			m.progress.MarkIncomplete(p.ID)
		} else {
			m.progress.MarkCompleted(p.ID)
		}
		m.progress.Save()

	default:
		if m.vpReady {
			var cmd tea.Cmd
			m.viewport, cmd = m.viewport.Update(msg)
			return m, cmd
		}
	}

	return m, nil
}

// --- Test Results Update ---

func (m Model) updateTestResults(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q":
		return m, tea.Quit

	case "esc":
		m.currentView = viewProblemDetail
		m.initDetailViewport()

	default:
		if m.testVPReady {
			var cmd tea.Cmd
			m.testVP, cmd = m.testVP.Update(msg)
			return m, cmd
		}
	}

	return m, nil
}

// --- Commands ---

func (m Model) openEditor(p problem.Problem) tea.Cmd {
	userFile := problem.UserCodePath(m.userCodeDir, p)

	// Ensure directory exists
	dir := filepath.Dir(userFile)
	os.MkdirAll(dir, 0755)

	// Create template if file doesn't exist
	if _, err := os.Stat(userFile); os.IsNotExist(err) {
		testsPath := filepath.Join(p.DirPath, "tests.json")
		template := problem.UserCodeTemplate(p, testsPath)
		os.WriteFile(userFile, []byte(template), 0644)
	}

	editor := m.config.ResolveEditor()

	c := exec.Command(editor, userFile)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		return editorFinishedMsg{err}
	})
}

func (m Model) runTestsCmd(p problem.Problem) tea.Cmd {
	userCodeDir := m.userCodeDir
	return func() tea.Msg {
		result, err := problem.RunTests(p, userCodeDir)
		return testResultMsg{result: result, err: err}
	}
}

func (m *Model) initDetailViewport() {
	h := m.height - 6
	if h < 4 {
		h = 4
	}
	m.viewport = viewport.New(m.width, h)
	m.viewport.SetContent(m.detailContent())
	m.vpReady = true
}

func (m *Model) initTestResultsViewport() {
	h := m.height - 6
	if h < 4 {
		h = 4
	}
	m.testVP = viewport.New(m.width, h)
	m.testVP.SetContent(m.testResultsContent())
	m.testVPReady = true
}

func (m Model) detailContent() string {
	if m.selectedIdx >= len(m.filtered) {
		return ""
	}
	p := m.filtered[m.selectedIdx]

	if m.showSolution {
		solPath := filepath.Join(p.DirPath, "solution.py")
		data, err := os.ReadFile(solPath)
		if err != nil {
			return styleError.Render("Could not read solution: " + err.Error())
		}
		return string(data)
	}

	return p.Readme
}

func (m Model) testResultsContent() string {
	if m.testResults == nil {
		return "No test results."
	}

	r := m.testResults
	var s string

	if r.Error != "" {
		s += styleError.Render("Error:") + "\n\n"
		s += r.Error + "\n"
		return s
	}

	s += fmt.Sprintf("Results: %d/%d passed", r.PassCnt, r.Total)
	if r.Passed {
		s += stylePass.Render("  ALL PASSED")
	}
	s += fmt.Sprintf("  (%s)\n\n", r.Elapsed.Round(1e6))

	for i, c := range r.Cases {
		if c.Passed {
			s += stylePass.Render(fmt.Sprintf("  PASS  %d. %s", i+1, c.Name)) + "\n"
		} else {
			s += styleFail.Render(fmt.Sprintf("  FAIL  %d. %s", i+1, c.Name)) + "\n"
			s += fmt.Sprintf("         expected: %s\n", c.Expected)
			s += fmt.Sprintf("              got: %s\n", c.Got)
		}
	}

	return s
}
