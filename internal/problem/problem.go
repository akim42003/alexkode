package problem

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type Problem struct {
	ID           string // e.g., "01-linear-algebra/01-matrix-transpose"
	Number       int    // problem number within category
	Title        string // e.g., "Matrix Transpose"
	Category     string // e.g., "Linear Algebra"
	CategorySlug string // e.g., "01-linear-algebra"
	Difficulty   string // "Easy", "Medium", "Hard"
	DirPath      string // absolute path to problem directory
	Readme       string // raw README.md content
}

var (
	reTitle      = regexp.MustCompile(`(?m)^#\s+(.+)$`)
	reDifficulty = regexp.MustCompile(`\*\*Difficulty:\*\*\s*(\w+)`)
)

func LoadProblems(baseDir string) ([]Problem, error) {
	var problems []Problem

	categories, err := os.ReadDir(baseDir)
	if err != nil {
		return nil, err
	}

	for _, cat := range categories {
		if !cat.IsDir() {
			continue
		}

		catSlug := cat.Name()
		categoryTitle := parseCategoryTitle(catSlug)

		probDirs, err := os.ReadDir(filepath.Join(baseDir, catSlug))
		if err != nil {
			continue
		}

		for _, prob := range probDirs {
			if !prob.IsDir() {
				continue
			}

			probDir := filepath.Join(baseDir, catSlug, prob.Name())
			readmeBytes, err := os.ReadFile(filepath.Join(probDir, "README.md"))
			if err != nil {
				continue
			}
			readme := string(readmeBytes)

			title := parseTitle(readme)
			difficulty := parseDifficulty(readme)
			number := parseNumber(prob.Name())

			problems = append(problems, Problem{
				ID:           filepath.Join(catSlug, prob.Name()),
				Number:       number,
				Title:        title,
				Category:     categoryTitle,
				CategorySlug: catSlug,
				Difficulty:   difficulty,
				DirPath:      probDir,
				Readme:       readme,
			})
		}
	}

	sort.Slice(problems, func(i, j int) bool {
		if problems[i].CategorySlug != problems[j].CategorySlug {
			return problems[i].CategorySlug < problems[j].CategorySlug
		}
		return problems[i].Number < problems[j].Number
	})

	return problems, nil
}

func Categories(problems []Problem) []string {
	seen := map[string]bool{}
	var cats []string
	for _, p := range problems {
		if !seen[p.Category] {
			seen[p.Category] = true
			cats = append(cats, p.Category)
		}
	}
	return cats
}

func FilterByCategory(problems []Problem, category string) []Problem {
	var out []Problem
	for _, p := range problems {
		if p.Category == category {
			out = append(out, p)
		}
	}
	return out
}

func FilterByDifficulty(problems []Problem, difficulty string) []Problem {
	var out []Problem
	for _, p := range problems {
		if p.Difficulty == difficulty {
			out = append(out, p)
		}
	}
	return out
}

func FilterByCompleted(problems []Problem, completed map[string]bool, wantCompleted bool) []Problem {
	var out []Problem
	for _, p := range problems {
		if completed[p.ID] == wantCompleted {
			out = append(out, p)
		}
	}
	return out
}

func parseTitle(readme string) string {
	m := reTitle.FindStringSubmatch(readme)
	if len(m) > 1 {
		return strings.TrimSpace(m[1])
	}
	return "Unknown"
}

func parseDifficulty(readme string) string {
	m := reDifficulty.FindStringSubmatch(readme)
	if len(m) > 1 {
		return strings.TrimSpace(m[1])
	}
	return "Unknown"
}

func parseNumber(slug string) int {
	parts := strings.SplitN(slug, "-", 2)
	if len(parts) > 0 {
		n, err := strconv.Atoi(parts[0])
		if err == nil {
			return n
		}
	}
	return 0
}

func parseCategoryTitle(slug string) string {
	parts := strings.SplitN(slug, "-", 2)
	if len(parts) < 2 {
		return slug
	}
	words := strings.Split(parts[1], "-")
	for i, w := range words {
		if len(w) > 0 {
			// Handle common short words
			if w == "nlp" || w == "cv" {
				words[i] = strings.ToUpper(w)
			} else if w == "and" || w == "of" || w == "the" {
				words[i] = w
			} else {
				words[i] = strings.ToUpper(w[:1]) + w[1:]
			}
		}
	}
	return strings.Join(words, " ")
}

func UserCodePath(userCodeDir string, p Problem) string {
	// Mirror problem directory structure: user_code/01-linear-algebra/01-matrix-transpose.py
	return filepath.Join(userCodeDir, p.CategorySlug, slugName(p.ID)+".py")
}

func UserCodeModule(p Problem) string {
	// Module name for Python import: "01-matrix-transpose" -> "01_matrix_transpose"
	name := slugName(p.ID)
	return strings.ReplaceAll(name, "-", "_")
}

func slugName(id string) string {
	// "01-linear-algebra/01-matrix-transpose" -> "01-matrix-transpose"
	return filepath.Base(id)
}

func UserCodeTemplate(p Problem, testsPath string) string {
	// Read tests.json to determine function signature
	sig := "data"
	testsBytes, err := os.ReadFile(testsPath)
	if err == nil {
		sig = inferSignature(string(testsBytes))
	}

	return "import numpy as np\n\n\ndef solution(" + sig + "):\n    # Your code here\n    pass\n"
}

func inferSignature(testsJSON string) string {
	// Quick heuristic: if test input contains { keys }, those are the param names
	// Look for "input": { pattern
	re := regexp.MustCompile(`"input"\s*:\s*\{([^}]+)\}`)
	m := re.FindStringSubmatch(testsJSON)
	if len(m) > 1 {
		// Extract key names
		keyRe := regexp.MustCompile(`"(\w+)"\s*:`)
		keys := keyRe.FindAllStringSubmatch(m[1], -1)
		var params []string
		for _, k := range keys {
			params = append(params, k[1])
		}
		if len(params) > 0 {
			return strings.Join(params, ", ")
		}
	}
	return "data"
}
