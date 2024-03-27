import pygame
from edificios import edificios
class Menu:
	def __init__(self,ancho,alto):self.ancho=ancho;self.alto=alto;self.menu_activo=False;self.opciones=list(edificios.keys());self.fuente=pygame.font.Font(None,30);self.indice_seleccionado=0;self.textos_opciones=[self.fuente.render(opcion,True,(180,180,180))for opcion in self.opciones]
	def toggle_menu(self):self.menu_activo=not self.menu_activo
	def dibujar(self,superficie):
		if self.menu_activo:
			sidebar_ancho=200;sidebar=pygame.Rect(self.ancho-sidebar_ancho,0,sidebar_ancho,self.alto);pygame.draw.rect(superficie,(70,70,70),sidebar)
			for(indice,opcion)in enumerate(self.opciones):edificio=edificios[opcion];color_texto=edificio.color if indice==self.indice_seleccionado else(180,180,180);texto=self.fuente.render(opcion,True,color_texto);superficie.blit(texto,(self.ancho-sidebar_ancho+10,30+indice*30))
	def manejar_evento(self,evento):
		if evento.type==pygame.KEYDOWN:
			if evento.key==pygame.K_UP and self.indice_seleccionado>0:self.indice_seleccionado-=1
			elif evento.key==pygame.K_DOWN and self.indice_seleccionado<len(self.opciones)-1:self.indice_seleccionado+=1
			elif evento.key==pygame.K_RETURN:edificio_seleccionado=self.opciones[self.indice_seleccionado];print(f"{edificio_seleccionado} seleccionado")