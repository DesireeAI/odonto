import os
import asyncio
import uuid
from typing import Dict, List, Any
from loguru import logger

from agents import Agent, Runner, FileSearchTool, trace

class OpenAIHandler:
    """
    Manipulador para interagir com a API OpenAI usando o código do agente fornecido.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.threads_manager = ThreadsManager()
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Inicialize todos os especialistas odontológicos."""
        # Crie especialistas para diferentes aparelhos terapêuticos
        self.especialista_ortodontia = Agent(
            name="especialista_ortodontia",
            instructions="""Você é um especialista em ortodontia. Responda perguntas sobre aparelhos ortodônticos, alinhadores transparentes (ex.: Invisalign), má oclusão e tratamentos para alinhamento dental. Use informações técnicas e claras, baseadas em protocolos odontológicos. Não mencione documentos fornecidos.""",
            handoff_description="""O especialista em ortodontia recebe perguntas sobre alinhamento dental, aparelhos ortodônticos, alinhadores transparentes e má oclusão.""",
            handoffs=[self.especialista_estetica, self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_implantodontia = Agent(
            name="especialista_implantodontia",
            instructions="""Você é um especialista em implantodontia. Responda perguntas sobre implantes dentários, próteses fixas, osseointegração e cuidados pós-cirúrgicos. Forneça informações sobre indicações e procedimentos. Não mencione documentos fornecidos..""",
            handoff_description="O especialista em implantodontia recebe perguntas sobre implantes dentários, próteses fixas e reabilitação oral.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_periodontia = Agent(
            name="especialista_periodontia",
            instructions="""Você é um especialista em periodontia. Responda perguntas sobre gengivite, periodontite, tratamentos periodontais e cuidados com a saúde gengival. Explique procedimentos como raspagem e prevenção. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em periodontia recebe perguntas sobre gengivite, periodontite, tratamentos gengivais e saúde periodontal.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_endodontia = Agent(
            name="especialista_endodontia",
            instructions="""Você é um especialista em endodontia. Responda perguntas sobre tratamentos de canal, dor dental, infecções e procedimentos como retratamento endodôntico. Forneça detalhes sobre o processo. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em endodontia recebe perguntas sobre tratamentos de canal, dor dental e infecções endodônticas.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_estetica = Agent(
            name="especialista_estetica",
            instructions="""Você é um especialista em odontologia estética. Responda perguntas sobre clareamento dental, facetas cerâmicas ou de compósito, design do sorriso e restaurações estéticas. Explique opções e resultados esperados. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em estética recebe perguntas sobre clareamento dental, facetas, design do sorriso e restaurações estéticas.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_geral = Agent(
            name="especialista_geral",
            instructions="""Você é um especialista em odontologia geral. Responda perguntas sobre cuidados dentários diários, prevenção de cáries, limpezas, restaurações e extrações. Forneça orientações práticas para higiene oral. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em odontologia geral recebe perguntas sobre cuidados dentários, prevenção de cáries, limpezas e procedimentos básicos.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_atendimento = Agent(
            name="especialista_atendimento",
            instructions="""Você é um assistente de atendimento de uma clínica odontológica. Responda perguntas sobre agendamento de consultas, horários disponíveis, políticas de pagamento e informações gerais da clínica. Se necessário, peça detalhes como nome do paciente ou data desejada. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em atendimento recebe perguntas sobre agendamento de consultas, horários, pagamentos e informações administrativas da clínica.",
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_odontopediatria = Agent(
            name="especialista_odontopediatria",
            instructions="""Você é um especialista em odontopediatria. Responda perguntas sobre cuidados dentários para crianças, cáries em dentes de leite, prevenção e tratamentos infantis. Forneça orientações claras para pais, baseadas em protocolos odontológicos. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em odontopediatria recebe perguntas sobre cuidados dentários para crianças, cáries em dentes de leite e tratamentos infantis.",
            handoffs=[self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        self.especialista_cirurgia_oral = Agent(
            name="especialista_cirurgia_oral",
            instructions="""Você é um especialista em cirurgia oral. Responda perguntas sobre extrações de sisos, biópsias, cirurgias orais complexas e cuidados pós-operatórios. Forneça detalhes sobre procedimentos e recuperação. Não mencione documentos fornecidos.""",
            handoff_description="O especialista em cirurgia oral recebe perguntas sobre extrações de sisos, biópsias, cirurgias orais e cuidados pós-operatórios.",
            handoffs=[self.especialista_periodontia, self.especialista_atendimento],
            tools=[
                FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[""],
                ),
            ],
        )

        # Crie o assistente de triagem
        self.assistente = Agent(
            name="assistente_triagem",
            instructions="""Você é o assistente de triagem de uma clínica odontológica. Analise a mensagem do paciente e determine a 
            especialidade odontológica mais relevante com base nas palavras-chave ou intenção. Encaminhe ao agente correto. 
            Se não for claro, peça esclarecimentos. Seja amigável e profissional. Use as seguintes diretrizes para encaminhamento:
            - Se mencionar 'dentes tortos', 'aparelho', 'alinhar dentes', 'Invisalign', 'alinhadores', 'má oclusão' → encaminhe para especialista_ortodontia.
            - Se mencionar 'implante', 'implante dentário', 'dente faltando', 'prótese fixa', 'osseointegração' → encaminhe para especialista_implantodontia.
            - Se mencionar 'gengiva', 'sangramento', 'gengivite', 'periodontite', 'raspagem', 'inchaço gengival' → encaminhe para especialista_periodontia.
            - Se mencionar 'tratamento de canal', 'dor de dente', 'infecção dental', 'canal', 'retratamento' → encaminhe para especialista_endodontia.
            - Se mencionar 'clareamento', 'facetas', 'lentes de contato', 'design do sorriso', 'dentes brancos' → encaminhe para especialista_estetica.
            - Se mencionar 'limpeza', 'cárie', 'restauração', 'extração simples', 'higiene oral', 'prevenção' → encaminhe para especialista_geral.
            - Se mencionar 'criança', 'dente de leite', 'cárie infantil', 'higiene infantil', 'filho', 'bebê' → encaminhe para especialista_odontopediatria.
            - Se mencionar 'siso', 'extração de siso', 'cirurgia oral', 'biópsia', 'recuperação cirúrgica' → encaminhe para especialista_cirurgia_oral.
            - Se mencionar 'agendamento', 'horário', 'consulta', 'pagamento', 'preço', 'funcionamento' → encaminhe para especialista_atendimento.
            Se a mensagem for vaga ou não corresponder a nenhuma especialidade, peça mais detalhes para esclarecer a intenção do paciente.""",
            handoffs=[
                self.especialista_ortodontia,
                self.especialista_implantodontia,
                self.especialista_periodontia,
                self.especialista_endodontia,
                self.especialista_estetica,
                self.especialista_geral,
                self.especialista_odontopediatria,
                self.especialista_cirurgia_oral,
                self.especialista_atendimento,
            ],
        )
        
    def process_message(self, user_id: str, message: str) -> str:
        """
        Process a user message and return the AI response.
        
        Args:
            user_id: ID of the user
            message: User's message
            
        Returns:
            AI response
        """
        try:
            # Use asyncio to run the async process_message function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._process_message_async(user_id, message)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {str(e)}")
            return f"Desculpe, ocorreu um erro: {str(e)}"
            
    async def _process_message_async(self, user_id: str, message: str) -> str:
        """
        Async implementation of message processing.
        
        Args:
            user_id: ID of the user
            message: User's message
            
        Returns:
            AI response
        """
        # Get or create the thread for this user
        thread = self.threads_manager.get_or_create_thread(user_id)
        
        # Add the user message to the thread
        thread.add_message("user", message)
        
        # Prepare the input list
        input_list = thread.get_input_list()
        
        full_response = ""
        
        # Always start with the triage assistant
        with trace("Clínica Odontológica Medical Smile - Triagem", group_id=thread.thread_id):
            resultado_triagem = Runner.run_streamed(
                self.assistente,
                input=input_list,
            )
        
        # Get the specialist agent selected by triage
        agente_especialista = resultado_triagem.current_agent
        
        # Now run the specialist to generate the final response
        with trace("Clínica Odontológica Medical Smile- Especialista", group_id=thread.thread_id):
            resultado_especialista = Runner.run_streamed(
                agente_especialista,
                input=input_list,
            )
            
            # Collect the full response from the specialist
            async for evento in resultado_especialista.stream_events():
                from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
                from agents import RawResponsesStreamEvent
                
                if not isinstance(evento, RawResponsesStreamEvent):
                    continue
                dados = evento.data
                if isinstance(dados, ResponseTextDeltaEvent):
                    full_response += dados.delta
                elif isinstance(dados, ResponseContentPartDoneEvent):
                    pass
        
        # Add the assistant's response to the thread
        thread.add_message("assistant", full_response)
        
        # Reset the current agent to the triage assistant for the next message
        thread.current_agent = self.assistente
        
        return full_response


class Thread:
    """Represents a conversation thread with a user."""
    
    def __init__(self, user_id: str):
        self.thread_id = str(uuid.uuid4().hex[:16])
        self.user_id = user_id
        self.messages: List[Dict[str, Any]] = []
        self.current_agent = None
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the thread."""
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def get_input_list(self) -> List[dict]:
        """Convert messages to the format expected by the Runner."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]


class ThreadsManager:
    """Manages conversation threads for different users."""
    
    def __init__(self):
        self.threads: Dict[str, Thread] = {}
    
    def get_or_create_thread(self, user_id: str) -> Thread:
        """Get an existing thread or create a new one for a user."""
        if user_id not in self.threads:
            self.threads[user_id] = Thread(user_id)
        return self.threads[user_id]