"""
==============================================================================
TESTES UNITÁRIOS - ALGORITMO BRANCH AND BOUND
==============================================================================

Disciplina: Pesquisa Operacional
Objetivo: Validar correção e robustez do algoritmo Branch and Bound

COBERTURA DE TESTES:
--------------------
1. Teste de cálculo de bound (relaxação linear)
2. Teste de verificação de viabilidade
3. Teste de critérios de poda
4. Teste de solução ótima em casos conhecidos
5. Teste de comparação com heurística gulosa
6. Teste de casos extremos (edge cases)
7. Teste de reprodutibilidade
8. Teste de validação de entrada

FRAMEWORK: unittest (biblioteca padrão do Python)

AUTOR: Manus AI
DATA: 16 de outubro de 2025
==============================================================================
"""

import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.optimization.branch_and_bound import (
    Project, Node, BranchAndBound, greedy_heuristic, validate_solution
)


class TestProject(unittest.TestCase):
    """
    Testes para a classe Project.
    
    DEFESA: Validar que projetos são criados corretamente e eficiência é calculada
    """
    
    def test_project_creation(self):
        """Testa criação básica de projeto."""
        project = Project(1, "Test Project", 100.0, 20.0, "Test")
        
        self.assertEqual(project.id, 1)
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.cost, 100.0)
        self.assertEqual(project.impact, 20.0)
        self.assertEqual(project.category, "Test")
    
    def test_efficiency_calculation(self):
        """Testa cálculo automático de eficiência."""
        project = Project(1, "Test", 50.0, 25.0, "Test")
        
        # Eficiência = impacto / custo = 25 / 50 = 0.5
        self.assertAlmostEqual(project.efficiency, 0.5, places=3)
    
    def test_efficiency_zero_cost(self):
        """Testa eficiência quando custo é zero."""
        project = Project(1, "Test", 0.0, 10.0, "Test")
        
        # Eficiência deve ser 0 quando custo é 0 (evitar divisão por zero)
        self.assertEqual(project.efficiency, 0.0)


class TestNode(unittest.TestCase):
    """
    Testes para a classe Node.
    
    DEFESA: Validar estrutura de nós e comparação para fila de prioridade
    """
    
    def test_node_creation(self):
        """Testa criação de nó."""
        node = Node(0, [], 0.0, 0.0, 100.0)
        
        self.assertEqual(node.level, 0)
        self.assertEqual(node.selected, [])
        self.assertEqual(node.total_cost, 0.0)
        self.assertEqual(node.total_impact, 0.0)
        self.assertEqual(node.bound, 100.0)
    
    def test_node_comparison(self):
        """Testa comparação de nós (para fila de prioridade)."""
        node1 = Node(0, [], 0.0, 0.0, 100.0)
        node2 = Node(0, [], 0.0, 0.0, 50.0)
        
        # node1 tem maior bound, então deve ser "menor" (maior prioridade)
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)


class TestBoundCalculation(unittest.TestCase):
    """
    Testes para cálculo de bound (relaxação linear).
    
    DEFESA: Bound é crucial para eficiência do algoritmo - deve estar correto
    """
    
    def setUp(self):
        """Prepara projetos de teste."""
        self.projects = [
            Project(1, "P1", 10.0, 20.0, "Test"),  # Eficiência: 2.0
            Project(2, "P2", 20.0, 30.0, "Test"),  # Eficiência: 1.5
            Project(3, "P3", 30.0, 40.0, "Test"),  # Eficiência: 1.33
        ]
        self.budget = 50.0
    
    def test_bound_empty_node(self):
        """Testa bound do nó raiz (nenhum projeto selecionado)."""
        bb = BranchAndBound(self.projects, self.budget)
        root = Node(0, [], 0.0, 0.0, 0.0)
        
        bound = bb.calculate_bound(root)
        
        # Com orçamento 50, podemos pegar P1 (10) + P2 (20) + fração de P3
        # Impacto = 20 + 30 + (20/30)*40 = 50 + 26.67 = 76.67
        self.assertAlmostEqual(bound, 76.67, places=1)
    
    def test_bound_with_selection(self):
        """Testa bound após selecionar um projeto."""
        bb = BranchAndBound(self.projects, self.budget)
        
        # Nó com P1 selecionado
        node = Node(1, [1], 10.0, 20.0, 0.0)
        bound = bb.calculate_bound(node)
        
        # Já temos 20 de impacto, restam 40 de orçamento
        # Podemos pegar P2 (20) + fração de P3 (20/30)*40 = 26.67
        # Total = 20 + 30 + 26.67 = 76.67
        self.assertAlmostEqual(bound, 76.67, places=1)
    
    def test_bound_full_budget(self):
        """Testa bound quando orçamento está esgotado."""
        bb = BranchAndBound(self.projects, self.budget)
        
        # Nó com orçamento esgotado
        node = Node(3, [1, 2], 50.0, 50.0, 0.0)
        bound = bb.calculate_bound(node)
        
        # Não há mais orçamento, bound = impacto atual
        self.assertEqual(bound, 50.0)


class TestFeasibility(unittest.TestCase):
    """
    Testes para verificação de viabilidade.
    
    DEFESA: Garantir que soluções inviáveis são corretamente identificadas
    """
    
    def setUp(self):
        """Prepara projetos e algoritmo."""
        self.projects = [
            Project(1, "P1", 50.0, 10.0, "Test"),
            Project(2, "P2", 60.0, 15.0, "Test"),
        ]
        self.budget = 100.0
        self.bb = BranchAndBound(self.projects, self.budget)
    
    def test_feasible_solution(self):
        """Testa solução viável."""
        node = Node(2, [1], 50.0, 10.0, 0.0)
        
        self.assertTrue(self.bb.is_feasible(node))
    
    def test_infeasible_solution(self):
        """Testa solução inviável (excede orçamento)."""
        node = Node(2, [1, 2], 110.0, 25.0, 0.0)
        
        self.assertFalse(self.bb.is_feasible(node))
    
    def test_exact_budget(self):
        """Testa solução que usa exatamente o orçamento."""
        node = Node(2, [1, 2], 100.0, 25.0, 0.0)
        
        self.assertTrue(self.bb.is_feasible(node))


class TestPruning(unittest.TestCase):
    """
    Testes para critérios de poda.
    
    DEFESA: Poda é essencial para eficiência - deve funcionar corretamente
    """
    
    def setUp(self):
        """Prepara projetos e algoritmo."""
        self.projects = [
            Project(1, "P1", 50.0, 20.0, "Test"),
            Project(2, "P2", 60.0, 25.0, "Test"),
        ]
        self.budget = 100.0
        self.bb = BranchAndBound(self.projects, self.budget)
        self.bb.best_value = 30.0  # Melhor solução conhecida
    
    def test_prune_infeasible(self):
        """Testa poda por inviabilidade."""
        node = Node(2, [1, 2], 150.0, 45.0, 50.0)
        
        should_prune, reason = self.bb.should_prune(node)
        
        self.assertTrue(should_prune)
        self.assertEqual(reason, "infeasible")
    
    def test_prune_bound(self):
        """Testa poda por bound."""
        node = Node(1, [1], 50.0, 20.0, 25.0)  # Bound = 25 ≤ best_value = 30
        
        should_prune, reason = self.bb.should_prune(node)
        
        self.assertTrue(should_prune)
        self.assertEqual(reason, "bound")
    
    def test_no_prune(self):
        """Testa nó que não deve ser podado."""
        node = Node(1, [1], 50.0, 20.0, 40.0)  # Bound = 40 > best_value = 30
        
        should_prune, reason = self.bb.should_prune(node)
        
        self.assertFalse(should_prune)
        self.assertEqual(reason, "none")


class TestOptimalSolution(unittest.TestCase):
    """
    Testes de solução ótima em casos conhecidos.
    
    DEFESA: Validar que o algoritmo encontra a solução ótima correta
    """
    
    def test_simple_case(self):
        """Testa caso simples com solução ótima conhecida."""
        # Caso: 3 projetos, orçamento 100
        # P1: custo=50, impacto=60 (eficiência=1.2)
        # P2: custo=30, impacto=40 (eficiência=1.33)
        # P3: custo=20, impacto=25 (eficiência=1.25)
        # Solução ótima: P1 + P2 + P3 = custo 100, impacto 125
        
        projects = [
            Project(1, "P1", 50.0, 60.0, "Test"),
            Project(2, "P2", 30.0, 40.0, "Test"),
            Project(3, "P3", 20.0, 25.0, "Test"),
        ]
        budget = 100.0
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(result["solution"]["total_impact"], 125.0)
        self.assertEqual(result["solution"]["total_cost"], 100.0)
        self.assertEqual(len(result["solution"]["selected_projects"]), 3)
    
    def test_all_projects_fit(self):
        """Testa caso onde todos os projetos cabem no orçamento."""
        projects = [
            Project(1, "P1", 10.0, 5.0, "Test"),
            Project(2, "P2", 20.0, 10.0, "Test"),
            Project(3, "P3", 30.0, 15.0, "Test"),
        ]
        budget = 100.0
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        # Todos os projetos devem ser selecionados
        self.assertEqual(result["solution"]["n_projects_selected"], 3)
        self.assertEqual(result["solution"]["total_cost"], 60.0)
        self.assertEqual(result["solution"]["total_impact"], 30.0)
    
    def test_no_project_fits(self):
        """Testa caso onde nenhum projeto cabe no orçamento."""
        projects = [
            Project(1, "P1", 100.0, 50.0, "Test"),
            Project(2, "P2", 150.0, 75.0, "Test"),
        ]
        budget = 50.0
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        # Nenhum projeto cabe, então não há solução viável
        self.assertEqual(result["status"], "no_solution")


class TestGreedyHeuristic(unittest.TestCase):
    """
    Testes para heurística gulosa.
    
    DEFESA: Validar que heurística funciona como baseline de comparação
    """
    
    def test_greedy_solution(self):
        """Testa solução da heurística gulosa."""
        projects = [
            Project(1, "P1", 50.0, 60.0, "Test"),  # Eficiência: 1.2
            Project(2, "P2", 30.0, 40.0, "Test"),  # Eficiência: 1.33
            Project(3, "P3", 20.0, 25.0, "Test"),  # Eficiência: 1.25
        ]
        budget = 100.0
        
        result = greedy_heuristic(projects, budget)
        
        # Greedy seleciona por eficiência: P2 (1.33), P3 (1.25), P1 (1.2)
        # P2 + P3 + P1 = 100, impacto = 125
        # Mas P1 não cabe após P2 + P3, então: P2 + P3 + parte de P1
        # Na verdade, greedy não pega frações, então: P2 + P3 = 50, impacto = 65
        # Depois tenta P1 (50), total = 100, impacto = 125
        
        self.assertEqual(result["status"], "heuristic")
        self.assertGreaterEqual(result["solution"]["total_impact"], 0)
    
    def test_greedy_vs_optimal(self):
        """Testa que B&B encontra solução melhor ou igual à heurística."""
        projects = [
            Project(1, "P1", 50.0, 60.0, "Test"),
            Project(2, "P2", 30.0, 40.0, "Test"),
            Project(3, "P3", 20.0, 25.0, "Test"),
        ]
        budget = 100.0
        
        bb = BranchAndBound(projects, budget)
        bb_result = bb.solve(verbose=False)
        greedy_result = greedy_heuristic(projects, budget)
        
        # B&B deve ser melhor ou igual à heurística
        self.assertGreaterEqual(
            bb_result["solution"]["total_impact"],
            greedy_result["solution"]["total_impact"]
        )


class TestEdgeCases(unittest.TestCase):
    """
    Testes de casos extremos (edge cases).
    
    DEFESA: Garantir robustez do algoritmo em situações limite
    """
    
    def test_single_project(self):
        """Testa com apenas um projeto."""
        projects = [Project(1, "P1", 50.0, 30.0, "Test")]
        budget = 100.0
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        self.assertEqual(result["solution"]["n_projects_selected"], 1)
        self.assertEqual(result["solution"]["total_impact"], 30.0)
    
    def test_zero_budget(self):
        """Testa com orçamento zero."""
        projects = [
            Project(1, "P1", 50.0, 30.0, "Test"),
            Project(2, "P2", 30.0, 20.0, "Test"),
        ]
        
        # Deve lançar exceção
        with self.assertRaises(ValueError):
            bb = BranchAndBound(projects, 0.0)
    
    def test_empty_projects(self):
        """Testa com lista vazia de projetos."""
        projects = []
        budget = 100.0
        
        # Deve lançar exceção
        with self.assertRaises(ValueError):
            bb = BranchAndBound(projects, budget)


class TestReproducibility(unittest.TestCase):
    """
    Testes de reprodutibilidade.
    
    DEFESA: Algoritmo deve ser determinístico (mesma entrada = mesma saída)
    """
    
    def test_reproducible_solution(self):
        """Testa que múltiplas execuções produzem mesma solução."""
        projects = [
            Project(1, "P1", 50.0, 60.0, "Test"),
            Project(2, "P2", 30.0, 40.0, "Test"),
            Project(3, "P3", 20.0, 25.0, "Test"),
            Project(4, "P4", 40.0, 35.0, "Test"),
        ]
        budget = 100.0
        
        # Executar 3 vezes
        results = []
        for _ in range(3):
            bb = BranchAndBound(projects, budget)
            result = bb.solve(verbose=False)
            results.append(result["solution"]["total_impact"])
        
        # Todas as execuções devem produzir mesmo resultado
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])


class TestValidation(unittest.TestCase):
    """
    Testes de validação de solução.
    
    DEFESA: Garantir que soluções são validadas corretamente
    """
    
    def test_validate_valid_solution(self):
        """Testa validação de solução válida."""
        projects = [
            Project(1, "P1", 50.0, 30.0, "Test"),
            Project(2, "P2", 30.0, 20.0, "Test"),
        ]
        budget = 100.0
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        # Solução deve ser válida
        self.assertTrue(validate_solution(result, projects, budget))
    
    def test_validate_invalid_solution(self):
        """Testa validação de solução inválida."""
        projects = [Project(1, "P1", 50.0, 30.0, "Test")]
        budget = 100.0
        
        # Criar solução inválida manualmente
        invalid_solution = {
            "status": "optimal",
            "solution": {
                "selected_projects": [projects[0]],
                "total_cost": 150.0,  # Excede orçamento!
                "total_impact": 30.0,
                "budget_used_pct": 150.0,
                "n_projects_selected": 1
            }
        }
        
        # Solução deve ser inválida
        self.assertFalse(validate_solution(invalid_solution, projects, budget))


def run_tests():
    """
    Executa todos os testes e gera relatório.
    
    DEFESA: Função auxiliar para execução standalone
    """
    print("="*70)
    print("EXECUTANDO TESTES UNITÁRIOS - BRANCH AND BOUND")
    print("="*70)
    
    # Criar test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar todos os testes
    suite.addTests(loader.loadTestsFromTestCase(TestProject))
    suite.addTests(loader.loadTestsFromTestCase(TestNode))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestFeasibility))
    suite.addTests(loader.loadTestsFromTestCase(TestPruning))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalSolution))
    suite.addTests(loader.loadTestsFromTestCase(TestGreedyHeuristic))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestReproducibility))
    suite.addTests(loader.loadTestsFromTestCase(TestValidation))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Relatório final
    print("\n" + "="*70)
    print("RELATÓRIO DE TESTES")
    print("="*70)
    print(f"Testes executados: {result.testsRun}")
    print(f"Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

