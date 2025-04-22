import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('ggplot')

colors = ['#4169E1', '#32CD32', '#FF8C00', '#9370DB', '#DC143C']
cmap_name = 'tsp_colors'
cm_tsp = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

class SimulatedAnnealing:
    def __init__(self, cities, temp_inicial=1000, taxa_resfriamento=0.995, temp_minima=1e-8, iteracoes_por_temp=100):
        self.cities = cities
        self.num_cities = len(cities)
        self.temp_inicial = temp_inicial
        self.temp_atual = temp_inicial
        self.taxa_resfriamento = taxa_resfriamento
        self.temp_minima = temp_minima
        self.iteracoes_por_temp = iteracoes_por_temp
        
        self.rota_atual = list(range(self.num_cities))
        random.shuffle(self.rota_atual)
        
        self.distancia_atual = self.calcular_distancia_total(self.rota_atual)
        self.distancia_inicial = self.distancia_atual
        
        self.melhor_rota = self.rota_atual.copy()
        self.melhor_distancia = self.distancia_atual
        
        self.historico_distancias = [self.distancia_atual]
        self.historico_temperaturas = [self.temp_atual]
        
        self.melhorias_por_fase = []  
        self.diferencas_distancias = []  
        self.iteracoes_por_fase = []  
        
        self.contador_aceitos = 0
        self.iteracoes_totais = 0
        self.melhorias_fase_atual = 0
        
        self.figuras = []
    
    def calcular_distancia(self, cidade1, cidade2):
        x1, y1 = self.cities[cidade1]
        x2, y2 = self.cities[cidade2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calcular_distancia_total(self, rota):
        distancia_total = 0
        for i in range(len(rota)):
            cidade_atual = rota[i]
            cidade_proxima = rota[(i + 1) % len(rota)]  
            distancia_total += self.calcular_distancia(cidade_atual, cidade_proxima)
        return distancia_total
    
    def gerar_vizinho(self):
        nova_rota = self.rota_atual.copy()
        
        i, j = random.sample(range(self.num_cities), 2)
        
        nova_rota[i], nova_rota[j] = nova_rota[j], nova_rota[i]
        
        return nova_rota
    
    def aceitar_solucao(self, nova_distancia):
        if nova_distancia < self.distancia_atual:
            self.melhorias_fase_atual += 1
            return True
        else:
            delta = nova_distancia - self.distancia_atual
            probabilidade = math.exp(-delta / self.temp_atual)
            
            if random.random() < probabilidade:
                self.diferencas_distancias.append(delta)
                return True
            return False
    
    def executar(self, visualizar=False):
        tempo_inicio = time.time()
        fase = 0
        
        if visualizar:
            fig = plt.figure(figsize=(15, 10))
            self.figuras.append(fig)
            plt.ion()  
        
        while self.temp_atual > self.temp_minima:
            fase += 1
            self.melhorias_fase_atual = 0
            iteracoes_fase = 0
            
            for _ in range(self.iteracoes_por_temp):
                self.iteracoes_totais += 1
                iteracoes_fase += 1

                nova_rota = self.gerar_vizinho()
                nova_distancia = self.calcular_distancia_total(nova_rota)
                
                if self.aceitar_solucao(nova_distancia):
                    self.rota_atual = nova_rota
                    self.distancia_atual = nova_distancia
                    self.contador_aceitos += 1
                    
                    if nova_distancia < self.melhor_distancia:
                        self.melhor_rota = nova_rota.copy()
                        self.melhor_distancia = nova_distancia
            
            self.historico_distancias.append(self.distancia_atual)
            self.historico_temperaturas.append(self.temp_atual)
            self.melhorias_por_fase.append(self.melhorias_fase_atual)
            self.iteracoes_por_fase.append(iteracoes_fase)
            
            self.temp_atual *= self.taxa_resfriamento

            if visualizar and fase % 10 == 0:
                self.visualizar_progresso_histogramas()
        
        tempo_total = time.time() - tempo_inicio
        
        return {
            'melhor_rota': self.melhor_rota,
            'melhor_distancia': self.melhor_distancia,
            'distancia_inicial': self.distancia_inicial,
            'melhoria_percentual': ((self.distancia_inicial - self.melhor_distancia) / self.distancia_inicial) * 100,
            'iteracoes_totais': self.iteracoes_totais,
            'solucoes_aceitas': self.contador_aceitos,
            'tempo_execucao': tempo_total,
            'fases_totais': fase
        }
    
    def visualizar_progresso_histogramas(self):
        plt.clf()
        
        plt.subplot(2, 2, 1)
        plt.title('Evolução da Distância', fontsize=14, fontweight='bold')
        
        n, bins, patches = plt.hist(self.historico_distancias, bins=30, alpha=0.8, 
                                     edgecolor='black', linewidth=1.2)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm_tsp(c))
            
        plt.xlabel('Distância', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.title('Melhorias por Fase', fontsize=14, fontweight='bold')
        
        n, bins, patches = plt.hist(self.melhorias_por_fase, bins=max(10, len(self.melhorias_por_fase)//10), 
                                    alpha=0.8, edgecolor='black', linewidth=1.2)
        

        for i, p in enumerate(patches):
            plt.setp(p, 'facecolor', cm_tsp(i/len(patches)))
            
        plt.xlabel('Número de Melhorias', fontsize=12)
        plt.ylabel('Frequência de Fases', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        if self.diferencas_distancias:
            plt.title('Diferenças de Distância em Aceitações Ruins', fontsize=14, fontweight='bold')
            
            n, bins, patches = plt.hist(self.diferencas_distancias, bins=30, alpha=0.8, 
                                        edgecolor='black', linewidth=1.2)

            for i, p in enumerate(patches):
                plt.setp(p, 'facecolor', cm_tsp(0.7 - i/len(patches)))
                
            plt.xlabel('Delta (Aumento na Distância)', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 4)
        x = [self.cities[i][0] for i in self.melhor_rota]
        y = [self.cities[i][1] for i in self.melhor_rota]
        
        x.append(x[0])
        y.append(y[0])
        
        plt.plot(x, y, '-', marker='o', linewidth=2, markersize=8, 
                color=colors[0], markerfacecolor=colors[2], markeredgecolor='black')
        plt.title(f'Melhor Rota (Distância: {self.melhor_distancia:.2f})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Coordenada X', fontsize=12)
        plt.ylabel('Coordenada Y', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

    def visualizar_histogramas_finais(self):
        fig = plt.figure(figsize=(15, 10))
        self.figuras.append(fig)
        
        plt.subplot(2, 2, 1)
        plt.title('Distribuição das Distâncias', fontsize=14, fontweight='bold')
        
        n, bins, patches = plt.hist(self.historico_distancias, bins=30, alpha=0.8, 
                                    edgecolor='black', linewidth=1.2)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm_tsp(c))
            
        plt.axvline(x=self.melhor_distancia, color='red', linestyle='--', linewidth=2,
                   label=f'Melhor: {self.melhor_distancia:.2f}')
        plt.legend(fontsize=10)
        plt.xlabel('Distância', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.title('Melhorias por Fase', fontsize=14, fontweight='bold')
        
        n, bins, patches = plt.hist(self.melhorias_por_fase, bins=max(10, len(self.melhorias_por_fase)//10), 
                                   alpha=0.8, edgecolor='black', linewidth=1.2)
        
        for i, p in enumerate(patches):
            plt.setp(p, 'facecolor', cm_tsp(i/len(patches)))
            
        plt.xlabel('Número de Melhorias', fontsize=12)
        plt.ylabel('Frequência de Fases', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        if self.diferencas_distancias:
            plt.title('Diferenças de Distância em Aceitações Ruins', fontsize=14, fontweight='bold')
            
            n, bins, patches = plt.hist(self.diferencas_distancias, bins=30, alpha=0.8, 
                                        edgecolor='black', linewidth=1.2)
            
            for i, p in enumerate(patches):
                plt.setp(p, 'facecolor', cm_tsp(0.7 - i/len(patches)))
                
            plt.xlabel('Delta (Aumento na Distância)', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 4)
        fases = range(len(self.historico_temperaturas))
        
        colors_temp = [cm_tsp(i/len(fases)) for i in fases]
        
        plt.bar(fases, self.historico_temperaturas, alpha=0.8, color=colors_temp, 
               edgecolor='black', linewidth=0.5)
        plt.title('Temperatura por Fase', fontsize=14, fontweight='bold')
        plt.xlabel('Fase', fontsize=12)
        plt.ylabel('Temperatura', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.draw()  

    def visualizar_rota_final(self, cidades, resultado):
        fig = plt.figure(figsize=(10, 8))
        self.figuras.append(fig)
        
        x = [cidades[i][0] for i in resultado['melhor_rota']]
        y = [cidades[i][1] for i in resultado['melhor_rota']]
        
        x.append(x[0])
        y.append(y[0])
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = mpl.collections.LineCollection(segments, cmap=cm_tsp)
        lc.set_array(np.linspace(0, 1, len(x)))
        lc.set_linewidth(3)
        
        plt.gca().add_collection(lc)
        
        plt.scatter(x[:-1], y[:-1], s=80, c=range(len(x)-1), cmap=cm_tsp, 
                   edgecolor='black', zorder=3)
        
        plt.scatter(x[0], y[0], s=150, c='red', marker='*', 
                  edgecolor='black', linewidth=1.5, zorder=5,
                  label='Cidade Inicial/Final')
        
        plt.title(f'Melhor Rota Final (Distância: {resultado["melhor_distancia"]:.2f})\n'
                 f'Melhoria: {resultado["melhoria_percentual"]:.2f}% em relação à rota inicial', 
                 fontsize=16, fontweight='bold')
        
        plt.xlabel('Coordenada X', fontsize=14)
        plt.ylabel('Coordenada Y', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        margin = 5
        plt.xlim(min(x) - margin, max(x) + margin)
        plt.ylim(min(y) - margin, max(y) + margin)
        
        for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):
            plt.text(xi, yi + 2, str(resultado['melhor_rota'][i]), 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        plt.draw() 
        
    def visualizar_convergencia(self):
        fig = plt.figure(figsize=(12, 8))
        self.figuras.append(fig)
        
        iterations = range(len(self.historico_distancias))
        
        plt.plot(iterations, self.historico_distancias, marker='', linewidth=2, 
                color=colors[0], alpha=0.9)
        
        melhorias = []
        for i in range(1, len(self.historico_distancias)):
            if self.historico_distancias[i] < self.historico_distancias[i-1]:
                melhorias.append(i)
        
        plt.plot(melhorias, [self.historico_distancias[i] for i in melhorias], 
                'o', color=colors[1], markersize=8)
        
        plt.axhline(y=self.melhor_distancia, color='red', linestyle='--', 
                   label=f'Melhor distância: {self.melhor_distancia:.2f}')
        
        plt.axhline(y=self.distancia_inicial, color='orange', linestyle='--', 
                   label=f'Distância inicial: {self.distancia_inicial:.2f}')
        
        plt.title('Convergência do Algoritmo de Têmpera Simulada', fontsize=16, fontweight='bold')
        plt.xlabel('Fase', fontsize=14)
        plt.ylabel('Distância Total da Rota', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        idx_melhor = self.historico_distancias.index(self.melhor_distancia)
        plt.annotate(f'Melhor solução\nFase {idx_melhor}',
                    xy=(idx_melhor, self.melhor_distancia),
                    xytext=(idx_melhor + len(iterations)*0.05, self.melhor_distancia + 20),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)
        
        plt.tight_layout()
        plt.draw()  

if __name__ == "__main__":
    np.random.seed(42)
    num_cidades = 100
    cidades = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(num_cidades)]
    
    tsp = SimulatedAnnealing(
        cities=cidades,
        temp_inicial=1000,
        taxa_resfriamento=0.995,
        temp_minima=1,
        iteracoes_por_temp=100
    )
    
    resultado = tsp.executar(visualizar=True)
    
    print("\nResultados da Têmpera Simulada para o TSP:")
    print(f"Distância inicial: {resultado['distancia_inicial']:.2f}")
    print(f"Melhor distância encontrada: {resultado['melhor_distancia']:.2f}")
    print(f"Melhoria: {resultado['melhoria_percentual']:.2f}%")
    print(f"Iterações totais: {resultado['iteracoes_totais']}")
    print(f"Soluções aceitas: {resultado['solucoes_aceitas']}")
    print(f"Taxa de aceitação: {resultado['solucoes_aceitas']/resultado['iteracoes_totais']*100:.2f}%")
    print(f"Fases totais: {resultado['fases_totais']}")
    print(f"Tempo de execução: {resultado['tempo_execucao']:.2f} segundos")
    
    tsp.visualizar_histogramas_finais()
    
    tsp.visualizar_rota_final(cidades, resultado)
    
    tsp.visualizar_convergencia()
    plt.ioff() 

    print("\nVisualizações foram geradas. Feche manualmente as janelas para encerrar o programa.")
    try:
        while True:
            plt.pause(0.1)  
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")