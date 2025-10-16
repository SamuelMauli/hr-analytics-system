"""
==============================================================================
IMPLEMENTAÇÃO DO ALGORITMO BRANCH AND BOUND
==============================================================================

Disciplina: Pesquisa Operacional
Problema: Seleção Ótima de Projetos de Retenção de Funcionários
Tipo: Problema da Mochila 0-1 (Knapsack Problem)

DESCRIÇÃO DO PROBLEMA:
----------------------
Uma empresa de RH possui um orçamento limitado para investir em projetos
de retenção de funcionários. Cada projeto tem um custo e um impacto esperado
na redução de rotatividade. O objetivo é selecionar o conjunto de projetos
que maximize o impacto total, respeitando o orçamento disponível.

MODELAGEM MATEMÁTICA:
---------------------
Variáveis de Decisão:
    x_i ∈ {0, 1}  onde i = 1, 2, ..., n
    x_i = 1 se o projeto i é selecionado
    x_i = 0 caso contrário

Função Objetivo (Maximização):
    max Z = Σ(i=1 até n) impacto_i * x_i

Restrições:
    Σ(i=1 até n) custo_i * x_i ≤ Orçamento
    x_i ∈ {0, 1} para todo i

ESTRATÉGIA DE BRANCH AND BOUND:
--------------------------------
1. Bound (Limite Superior): Relaxação Linear Fracionária
   - Permite selecionar frações de projetos
   - Ordena projetos por razão impacto/custo (eficiência)
   - Preenche mochila até o limite, permitindo fração no último item

2. Branching (Ramificação):
   - Para cada nó, cria dois ramos:
     * Ramo esquerdo: inclui o projeto (x_i = 1)
     * Ramo direito: exclui o projeto (x_i = 0)

3. Pruning (Poda):
   - Poda por inviabilidade: custo excede orçamento
   - Poda por otimalidade: bound ≤ melhor solução conhecida
   - Poda por completude: todos os projetos foram decididos

4. Busca:
   - Estratégia: Best-First Search (melhor bound primeiro)
   - Estrutura: Fila de prioridade (heap)

AUTOR: Manus AI
DATA: 16 de outubro de 2025
==============================================================================
"""

import heapq
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Project:
    """
    Representa um projeto de retenção de funcionários.
    
    Atributos:
        id: Identificador único do projeto
        name: Nome descritivo do projeto
        cost: Custo de implementação (em milhares de reais)
        impact: Impacto esperado na redução de rotatividade (%)
        category: Categoria do projeto (ex: Treinamento, Benefícios, etc.)
        efficiency: Razão impacto/custo (calculado automaticamente)
    """
    id: int
    name: str
    cost: float
    impact: float
    category: str
    efficiency: float = field(init=False)
    
    def __post_init__(self):
        """
        Calcula a eficiência do projeto após inicialização.
        Eficiência = impacto / custo (quanto maior, melhor)
        """
        if self.cost > 0:
            self.efficiency = self.impact / self.cost
        else:
            self.efficiency = 0.0


@dataclass
class Node:
    """
    Representa um nó na árvore de busca do Branch and Bound.
    
    Atributos:
        level: Nível do nó na árvore (profundidade)
        selected: Lista de projetos selecionados até este nó
        total_cost: Custo total acumulado dos projetos selecionados
        total_impact: Impacto total acumulado dos projetos selecionados
        bound: Limite superior estimado (melhor caso possível)
    
    DEFESA DE CÓDIGO:
    - Usamos @dataclass para reduzir boilerplate e melhorar legibilidade
    - O método __lt__ é necessário para comparação na fila de prioridade
    - Ordenamos por bound decrescente (maior bound = maior prioridade)
    """
    level: int
    selected: List[int]
    total_cost: float
    total_impact: float
    bound: float
    
    def __lt__(self, other):
        """
        Comparador para fila de prioridade.
        Nós com maior bound têm maior prioridade (best-first search).
        """
        return self.bound > other.bound


class BranchAndBound:
    """
    Implementação do algoritmo Branch and Bound para o problema da mochila 0-1.
    
    DEFESA DE CÓDIGO:
    -----------------
    1. Modularização: Cada método tem responsabilidade única e bem definida
    2. Documentação: Todos os métodos possuem docstrings explicativas
    3. Métricas: Rastreamos nós expandidos, podas e tempo de execução
    4. Reprodutibilidade: Algoritmo determinístico (mesma entrada = mesma saída)
    5. Eficiência: Uso de heap para fila de prioridade (O(log n) por operação)
    """
    
    def __init__(self, projects: List[Project], budget: float):
        """
        Inicializa o algoritmo Branch and Bound.
        
        Args:
            projects: Lista de projetos disponíveis
            budget: Orçamento total disponível (em milhares de reais)
        
        DEFESA: Validamos entradas para evitar erros em tempo de execução
        """
        if not projects:
            raise ValueError("Lista de projetos não pode estar vazia")
        if budget <= 0:
            raise ValueError("Orçamento deve ser positivo")
        
        self.projects = projects
        self.budget = budget
        self.n_projects = len(projects)
        
        # Ordenar projetos por eficiência (impacto/custo) em ordem decrescente
        # DEFESA: Ordenação é crucial para a qualidade do bound
        self.projects.sort(key=lambda p: p.efficiency, reverse=True)
        
        # Métricas de execução (para análise de desempenho)
        self.nodes_expanded = 0
        self.nodes_pruned_infeasible = 0
        self.nodes_pruned_bound = 0
        self.max_depth = 0
        self.execution_time = 0.0
        
        # Melhor solução encontrada
        self.best_solution: Optional[Node] = None
        self.best_value = 0.0
    
    def calculate_bound(self, node: Node) -> float:
        """
        Calcula o limite superior (bound) usando relaxação linear fracionária.
        
        Estratégia:
        1. Soma o impacto dos projetos já selecionados
        2. Para projetos restantes (ordenados por eficiência):
           - Se cabe inteiro, adiciona impacto completo
           - Se não cabe inteiro, adiciona fração proporcional
        
        Args:
            node: Nó atual da árvore de busca
        
        Returns:
            Limite superior do impacto possível a partir deste nó
        
        DEFESA DE CÓDIGO:
        -----------------
        - Complexidade: O(n) onde n é o número de projetos restantes
        - Relaxação linear: permite frações, fornecendo upper bound válido
        - Greedy approach: preenche por ordem de eficiência (ótimo para relaxação)
        """
        # Começamos com o impacto já acumulado
        bound = node.total_impact
        
        # Capacidade restante da "mochila"
        remaining_budget = self.budget - node.total_cost
        
        # Percorrer projetos restantes (não decididos ainda)
        for i in range(node.level, self.n_projects):
            project = self.projects[i]
            
            # Se o projeto cabe inteiro
            if project.cost <= remaining_budget:
                bound += project.impact
                remaining_budget -= project.cost
            else:
                # Adiciona fração proporcional (relaxação linear)
                # DEFESA: Esta é a chave da relaxação - permite frações
                fraction = remaining_budget / project.cost
                bound += project.impact * fraction
                break  # Não há mais orçamento disponível
        
        return bound
    
    def is_feasible(self, node: Node) -> bool:
        """
        Verifica se um nó representa uma solução viável.
        
        Args:
            node: Nó a ser verificado
        
        Returns:
            True se o custo total não excede o orçamento, False caso contrário
        
        DEFESA: Verificação simples mas essencial para garantir viabilidade
        """
        return node.total_cost <= self.budget
    
    def should_prune(self, node: Node) -> Tuple[bool, str]:
        """
        Determina se um nó deve ser podado (não expandido).
        
        Critérios de Poda:
        1. Inviabilidade: custo excede orçamento
        2. Otimalidade: bound não pode melhorar a melhor solução conhecida
        
        Args:
            node: Nó a ser avaliado
        
        Returns:
            Tupla (deve_podar, motivo)
        
        DEFESA DE CÓDIGO:
        -----------------
        - Poda por inviabilidade: evita explorar ramos impossíveis
        - Poda por otimalidade: evita explorar ramos subótimos
        - Retornamos o motivo para fins de logging e análise
        """
        # Poda por inviabilidade
        if not self.is_feasible(node):
            self.nodes_pruned_infeasible += 1
            return True, "infeasible"
        
        # Poda por otimalidade (bound não pode melhorar o melhor conhecido)
        if node.bound <= self.best_value:
            self.nodes_pruned_bound += 1
            return True, "bound"
        
        return False, "none"
    
    def update_best_solution(self, node: Node):
        """
        Atualiza a melhor solução encontrada se o nó atual for melhor.
        
        Args:
            node: Nó candidato a melhor solução
        
        DEFESA: Mantemos sempre a melhor solução viável encontrada
        """
        if self.is_feasible(node) and node.total_impact > self.best_value:
            self.best_solution = node
            self.best_value = node.total_impact
    
    def solve(self, verbose: bool = True) -> Dict:
        """
        Executa o algoritmo Branch and Bound para encontrar a solução ótima.
        
        Algoritmo:
        1. Inicializa fila de prioridade com nó raiz
        2. Enquanto a fila não estiver vazia:
           a. Remove nó com melhor bound (best-first search)
           b. Se deve podar, descarta o nó
           c. Se é folha viável, atualiza melhor solução
           d. Senão, expande o nó criando dois filhos:
              - Filho esquerdo: inclui próximo projeto
              - Filho direito: exclui próximo projeto
        3. Retorna melhor solução encontrada
        
        Args:
            verbose: Se True, imprime progresso durante execução
        
        Returns:
            Dicionário com solução, métricas e detalhes da execução
        
        DEFESA DE CÓDIGO:
        -----------------
        - Estrutura de dados: heap (fila de prioridade) para eficiência
        - Estratégia: best-first search (explora nós mais promissores primeiro)
        - Completude: garante encontrar solução ótima se existir
        - Otimalidade: bound garante que não perdemos a solução ótima
        - Complexidade: O(2^n) no pior caso, mas podas reduzem drasticamente
        """
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print("INICIANDO BRANCH AND BOUND")
            print("="*70)
            print(f"Número de projetos: {self.n_projects}")
            print(f"Orçamento disponível: R$ {self.budget:.2f}k")
            print(f"Estratégia de busca: Best-First (maior bound primeiro)")
            print("="*70)
        
        # Criar nó raiz (nenhum projeto selecionado ainda)
        root = Node(
            level=0,
            selected=[],
            total_cost=0.0,
            total_impact=0.0,
            bound=self.calculate_bound(Node(0, [], 0.0, 0.0, 0.0))
        )
        
        # Fila de prioridade (heap) - nós com maior bound têm prioridade
        # DEFESA: heapq é eficiente (O(log n)) e nativo do Python
        priority_queue = []
        heapq.heappush(priority_queue, root)
        
        # Loop principal do Branch and Bound
        while priority_queue:
            # Remove nó com melhor bound (best-first search)
            current_node = heapq.heappop(priority_queue)
            self.nodes_expanded += 1
            
            # Atualizar profundidade máxima alcançada
            self.max_depth = max(self.max_depth, current_node.level)
            
            # Verificar se deve podar este nó
            should_prune, prune_reason = self.should_prune(current_node)
            if should_prune:
                if verbose and self.nodes_expanded % 100 == 0:
                    print(f"Nó {self.nodes_expanded}: Podado ({prune_reason})")
                continue
            
            # Se chegamos a uma folha (todos os projetos foram decididos)
            if current_node.level == self.n_projects:
                self.update_best_solution(current_node)
                if verbose:
                    print(f"Solução viável encontrada: Impacto = {current_node.total_impact:.2f}%")
                continue
            
            # Expansão do nó: criar dois filhos
            # DEFESA: Branching é a essência do algoritmo - explorar ambas as decisões
            
            # Filho 1: INCLUIR o próximo projeto (x_i = 1)
            project = self.projects[current_node.level]
            left_child = Node(
                level=current_node.level + 1,
                selected=current_node.selected + [project.id],
                total_cost=current_node.total_cost + project.cost,
                total_impact=current_node.total_impact + project.impact,
                bound=0.0  # Será calculado abaixo
            )
            left_child.bound = self.calculate_bound(left_child)
            
            # Adicionar filho esquerdo à fila se não deve ser podado
            if not self.should_prune(left_child)[0]:
                heapq.heappush(priority_queue, left_child)
            
            # Filho 2: EXCLUIR o próximo projeto (x_i = 0)
            right_child = Node(
                level=current_node.level + 1,
                selected=current_node.selected[:],  # Cópia da lista
                total_cost=current_node.total_cost,
                total_impact=current_node.total_impact,
                bound=0.0  # Será calculado abaixo
            )
            right_child.bound = self.calculate_bound(right_child)
            
            # Adicionar filho direito à fila se não deve ser podado
            if not self.should_prune(right_child)[0]:
                heapq.heappush(priority_queue, right_child)
            
            # Log de progresso a cada 100 nós
            if verbose and self.nodes_expanded % 100 == 0:
                print(f"Nós expandidos: {self.nodes_expanded} | "
                      f"Fila: {len(priority_queue)} | "
                      f"Melhor: {self.best_value:.2f}%")
        
        # Calcular tempo de execução
        self.execution_time = time.time() - start_time
        
        if verbose:
            print("="*70)
            print("EXECUÇÃO CONCLUÍDA")
            print("="*70)
        
        # Preparar resultado
        result = self._prepare_result()
        
        if verbose:
            self._print_summary(result)
        
        return result
    
    def _prepare_result(self) -> Dict:
        """
        Prepara o dicionário de resultado com solução e métricas.
        
        Returns:
            Dicionário contendo solução ótima e estatísticas de execução
        
        DEFESA: Estrutura clara e completa para análise e visualização
        """
        if self.best_solution is None:
            return {
                "status": "no_solution",
                "message": "Nenhuma solução viável encontrada",
                "metrics": self._get_metrics()
            }
        
        # Recuperar projetos selecionados
        selected_projects = [
            p for p in self.projects if p.id in self.best_solution.selected
        ]
        
        return {
            "status": "optimal",
            "solution": {
                "selected_projects": selected_projects,
                "total_cost": self.best_solution.total_cost,
                "total_impact": self.best_solution.total_impact,
                "budget_used_pct": (self.best_solution.total_cost / self.budget) * 100,
                "n_projects_selected": len(selected_projects)
            },
            "metrics": self._get_metrics(),
            "details": {
                "budget": self.budget,
                "n_projects_available": self.n_projects,
                "execution_time": self.execution_time
            }
        }
    
    def _get_metrics(self) -> Dict:
        """
        Retorna métricas de execução do algoritmo.
        
        Returns:
            Dicionário com estatísticas de desempenho
        
        DEFESA: Métricas essenciais para avaliar eficiência e qualidade
        """
        total_pruned = self.nodes_pruned_infeasible + self.nodes_pruned_bound
        
        return {
            "nodes_expanded": self.nodes_expanded,
            "nodes_pruned_total": total_pruned,
            "nodes_pruned_infeasible": self.nodes_pruned_infeasible,
            "nodes_pruned_bound": self.nodes_pruned_bound,
            "max_depth": self.max_depth,
            "execution_time_seconds": self.execution_time,
            "pruning_efficiency_pct": (total_pruned / (self.nodes_expanded + total_pruned) * 100) 
                                      if (self.nodes_expanded + total_pruned) > 0 else 0
        }
    
    def _print_summary(self, result: Dict):
        """
        Imprime resumo formatado da execução.
        
        Args:
            result: Dicionário de resultado do algoritmo
        
        DEFESA: Saída clara e informativa para o usuário
        """
        print("\n📊 SOLUÇÃO ÓTIMA ENCONTRADA")
        print("-" * 70)
        
        if result["status"] == "optimal":
            sol = result["solution"]
            print(f"Impacto Total: {sol['total_impact']:.2f}%")
            print(f"Custo Total: R$ {sol['total_cost']:.2f}k")
            print(f"Orçamento Utilizado: {sol['budget_used_pct']:.1f}%")
            print(f"Projetos Selecionados: {sol['n_projects_selected']}/{self.n_projects}")
            
            print("\n📋 PROJETOS SELECIONADOS:")
            print("-" * 70)
            for proj in sol['selected_projects']:
                print(f"  • {proj.name}")
                print(f"    Custo: R$ {proj.cost:.2f}k | "
                      f"Impacto: {proj.impact:.2f}% | "
                      f"Eficiência: {proj.efficiency:.3f}")
        
        print("\n📈 MÉTRICAS DE EXECUÇÃO")
        print("-" * 70)
        metrics = result["metrics"]
        print(f"Nós Expandidos: {metrics['nodes_expanded']}")
        print(f"Nós Podados (Total): {metrics['nodes_pruned_total']}")
        print(f"  - Por Inviabilidade: {metrics['nodes_pruned_infeasible']}")
        print(f"  - Por Bound: {metrics['nodes_pruned_bound']}")
        print(f"Profundidade Máxima: {metrics['max_depth']}")
        print(f"Eficiência de Poda: {metrics['pruning_efficiency_pct']:.1f}%")
        print(f"Tempo de Execução: {metrics['execution_time_seconds']:.3f}s")
        print("=" * 70)


def greedy_heuristic(projects: List[Project], budget: float) -> Dict:
    """
    Heurística gulosa para comparação com Branch and Bound.
    
    Estratégia:
    - Ordena projetos por eficiência (impacto/custo) decrescente
    - Seleciona projetos enquanto houver orçamento disponível
    
    Args:
        projects: Lista de projetos disponíveis
        budget: Orçamento total disponível
    
    Returns:
        Dicionário com solução heurística e métricas
    
    DEFESA DE CÓDIGO:
    -----------------
    - Complexidade: O(n log n) devido à ordenação
    - Garantia: Não garante solução ótima, mas é rápida
    - Uso: Baseline para comparação com Branch and Bound
    """
    start_time = time.time()
    
    # Ordenar por eficiência (greedy choice)
    sorted_projects = sorted(projects, key=lambda p: p.efficiency, reverse=True)
    
    selected = []
    total_cost = 0.0
    total_impact = 0.0
    
    # Selecionar projetos enquanto couber no orçamento
    for project in sorted_projects:
        if total_cost + project.cost <= budget:
            selected.append(project)
            total_cost += project.cost
            total_impact += project.impact
    
    execution_time = time.time() - start_time
    
    return {
        "status": "heuristic",
        "solution": {
            "selected_projects": selected,
            "total_cost": total_cost,
            "total_impact": total_impact,
            "budget_used_pct": (total_cost / budget) * 100,
            "n_projects_selected": len(selected)
        },
        "metrics": {
            "execution_time_seconds": execution_time
        }
    }


# ==============================================================================
# FUNÇÕES AUXILIARES PARA TESTES E VALIDAÇÃO
# ==============================================================================

def validate_solution(solution: Dict, projects: List[Project], budget: float) -> bool:
    """
    Valida se uma solução é viável.
    
    Args:
        solution: Dicionário de solução
        projects: Lista de projetos
        budget: Orçamento disponível
    
    Returns:
        True se a solução é viável, False caso contrário
    
    DEFESA: Validação essencial para garantir correção do algoritmo
    """
    if solution["status"] != "optimal":
        return False
    
    sol = solution["solution"]
    
    # Verificar se custo não excede orçamento
    if sol["total_cost"] > budget:
        return False
    
    # Verificar se impacto está correto
    calculated_impact = sum(p.impact for p in sol["selected_projects"])
    if abs(calculated_impact - sol["total_impact"]) > 0.01:
        return False
    
    return True


if __name__ == "__main__":
    """
    Exemplo de uso do algoritmo Branch and Bound.
    
    DEFESA: Código de exemplo para demonstração e testes
    """
    # Criar projetos de exemplo
    example_projects = [
        Project(1, "Programa de Mentoria", 50, 15, "Desenvolvimento"),
        Project(2, "Aumento Salarial Geral", 200, 25, "Compensação"),
        Project(3, "Home Office Flexível", 30, 12, "Benefícios"),
        Project(4, "Treinamento Técnico", 80, 18, "Desenvolvimento"),
        Project(5, "Plano de Carreira", 40, 20, "Desenvolvimento"),
    ]
    
    budget = 150  # R$ 150k
    
    print("Executando Branch and Bound...")
    bb = BranchAndBound(example_projects, budget)
    result = bb.solve(verbose=True)
    
    print("\nExecutando Heurística Gulosa...")
    heuristic_result = greedy_heuristic(example_projects, budget)
    
    print("\n📊 COMPARAÇÃO: Branch and Bound vs Heurística")
    print("="*70)
    print(f"B&B Impacto: {result['solution']['total_impact']:.2f}%")
    print(f"Heurística Impacto: {heuristic_result['solution']['total_impact']:.2f}%")
    print(f"Diferença: {result['solution']['total_impact'] - heuristic_result['solution']['total_impact']:.2f}%")

