package problem

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type TestResult struct {
	Passed  bool
	Total   int
	PassCnt int
	Cases   []CaseResult
	Error   string
	Elapsed time.Duration
}

type CaseResult struct {
	Name     string `json:"name"`
	Passed   bool   `json:"passed"`
	Got      string `json:"got"`
	Expected string `json:"expected"`
}

func RunTests(p Problem, userCodeDir string) (*TestResult, error) {
	userFile := UserCodePath(userCodeDir, p)
	if _, err := os.Stat(userFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("no user code found — press 'e' to edit first")
	}

	testsPath := filepath.Join(p.DirPath, "tests.json")
	testsBytes, err := os.ReadFile(testsPath)
	if err != nil {
		return nil, fmt.Errorf("cannot read tests.json: %w", err)
	}

	moduleName := UserCodeModule(p)
	codeDir := filepath.Dir(userFile)

	script := generateTestScript(string(testsBytes), moduleName, codeDir)

	tmpFile, err := os.CreateTemp("", "deepml_test_*.py")
	if err != nil {
		return nil, fmt.Errorf("cannot create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(script); err != nil {
		tmpFile.Close()
		return nil, fmt.Errorf("cannot write test script: %w", err)
	}
	tmpFile.Close()

	start := time.Now()
	cmd := exec.Command("python3", tmpFile.Name())
	output, err := cmd.CombinedOutput()
	elapsed := time.Since(start)

	outStr := strings.TrimSpace(string(output))

	if err != nil {
		// Python error — show the output as the error
		return &TestResult{
			Passed:  false,
			Error:   outStr,
			Elapsed: elapsed,
		}, nil
	}

	var cases []CaseResult
	if err := json.Unmarshal([]byte(outStr), &cases); err != nil {
		return &TestResult{
			Passed:  false,
			Error:   "failed to parse test output:\n" + outStr,
			Elapsed: elapsed,
		}, nil
	}

	passCnt := 0
	allPassed := true
	for _, c := range cases {
		if c.Passed {
			passCnt++
		} else {
			allPassed = false
		}
	}

	return &TestResult{
		Passed:  allPassed,
		Total:   len(cases),
		PassCnt: passCnt,
		Cases:   cases,
		Elapsed: elapsed,
	}, nil
}

func generateTestScript(testsJSON, moduleName, codeDir string) string {
	// Escape backticks and quotes in tests JSON for embedding
	escaped := strings.ReplaceAll(testsJSON, `\`, `\\`)
	escaped = strings.ReplaceAll(escaped, `'`, `\'`)

	return fmt.Sprintf(`import json
import sys
import traceback
import numpy as np

sys.path.insert(0, %q)

from %s import solution

tests_json = '''%s'''
tests = json.loads(tests_json)

results = []

def to_comparable(val):
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, list):
        try:
            return np.array(val, dtype=float)
        except (ValueError, TypeError):
            return val
    return val

for tc in tests['test_cases']:
    try:
        inp = tc['input']

        if isinstance(inp, dict):
            kwargs = {}
            for k, v in inp.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    kwargs[k] = np.array(v, dtype=float)
                else:
                    kwargs[k] = v
            got = solution(**kwargs)
        elif isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list):
            got = solution(np.array(inp, dtype=float))
        elif isinstance(inp, list):
            got = solution(np.array(inp, dtype=float))
        elif isinstance(inp, str):
            got = solution(inp)
        else:
            got = solution(inp)

        exp = tc['expected']

        if isinstance(got, np.ndarray):
            exp_arr = to_comparable(exp)
            if isinstance(exp_arr, np.ndarray):
                passed = np.allclose(got, exp_arr, rtol=1e-5, atol=1e-8)
            else:
                passed = False
            got_str = str(got.tolist())
        elif isinstance(got, dict):
            passed = True
            if isinstance(exp, dict):
                for k in exp:
                    if k not in got:
                        passed = False
                        break
                    if isinstance(exp[k], (int, float)) and isinstance(got[k], (int, float)):
                        if abs(got[k] - exp[k]) > 1e-4:
                            passed = False
                            break
                    elif got[k] != exp[k]:
                        passed = False
                        break
            else:
                passed = False
            got_str = json.dumps(got, default=str)
        elif isinstance(got, (list, tuple)):
            exp_c = to_comparable(exp)
            got_c = to_comparable(got)
            if isinstance(got_c, np.ndarray) and isinstance(exp_c, np.ndarray):
                try:
                    passed = np.allclose(got_c, exp_c, rtol=1e-5, atol=1e-8)
                except Exception:
                    passed = False
            else:
                passed = got == exp
            got_str = json.dumps(got, default=str) if not isinstance(got, str) else got
        elif isinstance(got, (int, float)):
            if isinstance(exp, (int, float)):
                passed = abs(got - exp) < 1e-5
            else:
                passed = got == exp
            got_str = str(got)
        else:
            passed = got == exp
            got_str = str(got)

        exp_str = json.dumps(exp, default=str) if not isinstance(exp, str) else exp
        results.append({"name": tc["name"], "passed": bool(passed), "got": got_str, "expected": exp_str})

    except Exception:
        results.append({
            "name": tc["name"],
            "passed": False,
            "got": traceback.format_exc(),
            "expected": json.dumps(tc.get("expected", ""), default=str)
        })

print(json.dumps(results))
`, codeDir, moduleName, escaped)
}
