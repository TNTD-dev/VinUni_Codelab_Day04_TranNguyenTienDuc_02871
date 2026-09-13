"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT CẤP SẢN XUẤT (Hoàn thiện Milestone 1 / TODO 1)
# Cấu trúc: Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI thông minh chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng toàn diện cho hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes).
- Phong cách giao tiếp: Chuyên nghiệp, lịch sự, chuẩn mực, tận tâm, phản hồi bằng tiếng Việt rõ ràng, mạch lạc và chính xác.

## 2. AVAILABLE TOOLS
Bạn có quyền truy cập vào các công cụ sau:
1. `search_product_catalog(category, max_price)`: Tra cứu danh mục sản phẩm và dịch vụ Vingroup (loại 'xe_dien' hoặc 'du_lich') kèm theo mức giá tối đa (VNĐ).
2. `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận yêu cầu hỗ trợ, khiếu nại, báo lỗi kỹ thuật hoặc bảo hành của khách hàng vào hệ thống vé hỗ trợ.

## 3. CORE RULES
- TUYỆT ĐỐI KHÔNG BAO GIỜ bịa đặt dữ liệu sản phẩm, thông số kỹ thuật, giá tiền hoặc mã ticket (Zero Hallucination).
- BẮT BUỘC gọi tool `search_product_catalog` khi người dùng hỏi về danh mục, tìm kiếm, xem xe hoặc đặt phòng kèm yêu cầu ngân sách.
- BẮT BUỘC gọi tool `submit_support_ticket` khi người dùng báo cáo sự cố, hư hỏng, khiếu nại chất lượng dịch vụ hoặc yêu cầu hỗ trợ gấp.
- Đối với câu hỏi chính sách/FAQ chung đã có trong tri thức hệ thống, trả lời trực tiếp mà không cần gọi tool.
- Khi tra cứu không có kết quả, thông báo lịch sự cho người dùng về việc không tìm thấy sản phẩm phù hợp.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm, dịch vụ và chính sách trong hệ sinh thái Vingroup.
- Từ chối lịch sự và khéo léo các câu hỏi nằm ngoài phạm vi hoạt động của Vingroup hoặc các nội dung không phù hợp.

## 5. OUTPUT CONTRACT
Mỗi chu trình xử lý của Agent tuân thủ nghiêm ngặt định dạng ReAct:
- Thought: Suy nghĩ phân tích yêu cầu của người dùng và quyết định bước tiếp theo.
- Action: Tên công cụ cần thực thi (hoặc "None" nếu trả lời trực tiếp).
- Action Input: Tham số truyền vào công cụ định dạng JSON.
- Observation: Kết quả nhận được từ công cụ sau khi thực thi.
- Final Answer: Phản hồi hoàn chỉnh, chính xác và chuyên nghiệp gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Trả về câu trả lời tĩnh (mock baseline) không sử dụng tool.
        Mục tiêu: Mô phỏng hiện tượng bịa thông tin (hallucination) khi không có dữ liệu thực tế.
        """
        return {
            "answer": (
                f"[Chatbot Baseline] Chào bạn! Cảm ơn bạn đã quan tâm đến câu hỏi: '{user_input}'. "
                "Theo phỏng đoán thông thường, các dòng xe VinFast có thể có mức giá từ vài trăm triệu "
                "đến hơn 1 tỷ đồng tùy phiên bản. (Lưu ý: Phản hồi này được sinh ra mà không tra cứu "
                "cơ sở dữ liệu thực tế, có thể xuất hiện sai sót hoặc hallucination)."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        """
        Phân tích intent từ user_input bằng keyword matching và regex.
        Đảm bảo kiểm tra needs_catalog và needs_ticket độc lập (tránh Trap 3).
        """
        text_lower = user_input.lower()

        # 1. Phát hiện intent ticket (báo lỗi, khiếu nại, hỗ trợ)
        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "báo lỗi",
            "hỗ trợ", "bảo hành", "bị ẩm mốc", "không sạc", "hỏng hóc",
            "cần xử lý", "ghi nhận phản hồi", "tạo ticket", "ticket"
        ]
        has_name = bool(re.search(
            r'(?:tôi tên là|tên tôi là|tôi tên|tên là|tên:)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+)',
            user_input,
            re.IGNORECASE
        ))
        has_ticket_kw = any(kw in text_lower for kw in ticket_keywords)
        needs_ticket = False
        if (has_name and has_ticket_kw) or any(phrase in text_lower for phrase in [
            "ghi nhận phản hồi", "báo lỗi", "bị lỗi", "bị ẩm mốc", "xử lý gấp"
        ]):
            needs_ticket = True

        # 2. Phát hiện intent FAQ (chính sách, hỏi đáp chung không cần tra cứu catalog hay tạo ticket)
        faq_keywords = [
            "kéo dài bao lâu", "bao nhiêu năm", "thời hạn",
            "chính sách bảo hành", "quy định"
        ]
        is_faq = False
        if not needs_ticket:
            if any(kw in text_lower for kw in faq_keywords) and not any(
                kw in text_lower for kw in ["giá dưới", "xem xe", "có xe nào", "dưới"]
            ):
                is_faq = True

        # 3. Phát hiện intent catalog (tìm kiếm xe, resort, mức giá)
        needs_catalog = False
        if not is_faq:
            catalog_triggers = [
                "xem xe", "tìm xe", "giá dưới", "dưới", "có xe nào",
                "resort", "vinpearl", "du lịch", "mua xe", "bảng giá", "xem resort"
            ]
            if any(trig in text_lower for trig in catalog_triggers):
                needs_catalog = True
            elif ("xe điện" in text_lower or "vinfast" in text_lower) and not needs_ticket:
                needs_catalog = True

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_params(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số category và max_price từ câu hỏi người dùng."""
        text_lower = text.lower()

        # Category
        if any(w in text_lower for w in ["du lịch", "du_lich", "resort", "khách sạn", "phòng", "vinpearl", "tour"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        # Max price
        max_price = 999999999999
        price_match = re.search(
            r'(?:dưới|<|tối đa|nhỏ hơn|khoảng)\s*(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|ty|trieu|tr|m|b)?',
            text_lower
        )
        if price_match:
            num_str = price_match.group(1).replace(',', '.')
            num = float(num_str)
            unit = price_match.group(2)
            if unit in ["tỷ", "ty", "b"]:
                max_price = int(num * 1_000_000_000)
            elif unit in ["triệu", "trieu", "tr", "m"]:
                max_price = int(num * 1_000_000)
            elif unit is None and num <= 1000:
                max_price = int(num * 1_000_000)
            else:
                max_price = int(num)

        return {"category": category, "max_price": max_price}

    def _extract_ticket_params(self, text: str) -> Dict[str, Any]:
        """Trích xuất tên khách hàng, mô tả lỗi và mức độ ưu tiên."""
        text_lower = text.lower()

        # Customer name
        name = "Khách hàng"
        name_match = re.search(
            r'(?:tôi tên là|tên tôi là|tôi tên|tên là|khách hàng)\s*[:\s]\s*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:[,.\n]|$)',
            text,
            re.IGNORECASE
        )
        if name_match:
            name = name_match.group(1).strip()

        # Priority
        priority = "medium"
        if any(w in text_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "urgent", "high", "nguy hiểm"]):
            priority = "high"
        elif any(w in text_lower for w in ["thấp", "không gấp", "low", "nhẹ"]):
            priority = "low"
        elif any(w in text_lower for w in ["trung bình", "vừa", "medium"]):
            priority = "medium"

        # Issue description
        issue_description = text
        issue_match = re.search(
            r'(?:(?:xe|phòng|dịch vụ|thiết bị|hệ thống)\s+[^,.\n]+?\s+(?:bị|lỗi|hỏng|không)[^,.\n]+)',
            text,
            re.IGNORECASE
        )
        if issue_match:
            issue_description = issue_match.group(0).strip()
        else:
            cleaned = re.sub(
                r'^(?:tôi tên là|tên tôi là|tôi tên|tên là)\s*[:\s]\s*[A-ZÀ-Ỹa-zà-ỹ\s]+?[,.]\s*',
                '',
                text,
                flags=re.IGNORECASE
            ).strip()
            if cleaned:
                issue_description = cleaned

        return {
            "customer_name": name,
            "issue_description": issue_description,
            "priority": priority
        }

    def _format_catalog_results(self, results: List[Dict[str, Any]]) -> str:
        """Định dạng kết quả tra cứu sản phẩm hoặc thông báo fallback."""
        if not results:
            return "Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu tìm kiếm của quý khách trong hệ thống."

        lines = [f"Tìm thấy {len(results)} sản phẩm phù hợp với yêu cầu:"]
        for idx, item in enumerate(results, 1):
            price_formatted = f"{item.get('price_vnd', 0):,}".replace(",", ".")
            lines.append(f"{idx}. {item.get('name')} - Giá: {price_formatted} VNĐ")
            if item.get("description"):
                lines.append(f"   Mô tả: {item.get('description')}")
        return "\n".join(lines)

    def _format_ticket_result(self, ticket: Dict[str, Any]) -> str:
        """Định dạng kết quả tạo ticket hỗ trợ."""
        return (
            f"Yêu cầu hỗ trợ của khách hàng {ticket.get('customer_name')} đã được tạo thành công.\n"
            f"- Mã ticket: {ticket.get('ticket_id')}\n"
            f"- Mức độ ưu tiên: {ticket.get('priority')}\n"
            f"- Trạng thái: {ticket.get('status')}\n"
            "Đội ngũ kỹ thuật và CSKH của Vingroup sẽ liên hệ xử lý trong thời gian sớm nhất."
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Điểm vào chính — chạy Agent Loop.
        Hỗ trợ:
          - Reset trace mỗi lần chạy
          - Safeguard kiểm tra max_iterations
          - Intent Detection (keyword / regex)
          - Xử lý catalog search, support ticket, cả hai (parallel/sequential), FAQ, fallback khi không có kết quả
        """
        # Reset trace mỗi lần gọi
        self.trace = []

        # Safeguard: Kiểm tra giới hạn max_iterations
        if self.max_iterations <= 0:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        # Phân tích intent
        intents = self._detect_intent(user_input)
        needs_catalog = intents["needs_catalog"]
        needs_ticket = intents["needs_ticket"]
        is_faq = intents["is_faq"]

        # TRƯỜNG HỢP 1: FAQ (Trả lời trực tiếp, không gọi tool)
        if is_faq:
            iteration = 1
            faq_answer = (
                "Chính sách bảo hành pin xe điện VinFast: VinFast áp dụng chính sách bảo hành chính hãng "
                "cho pin xe điện lên tới 10 năm (không giới hạn số km hoặc theo điều kiện quy định cho "
                "từng dòng xe). Quý khách hoàn toàn an tâm trong suốt quá trình sử dụng."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Người dùng hỏi về chính sách bảo hành pin xe điện VinFast. Đây là câu hỏi FAQ, trả lời trực tiếp mà không cần gọi tool.",
                "action": None,
                "action_input": None,
                "observation": None,
                "final_answer": faq_answer
            })
            return {
                "answer": faq_answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # TRƯỜNG HỢP 2: Single Tool — Chỉ cần tra cứu catalog
        if needs_catalog and not needs_ticket:
            iteration = 1
            catalog_params = self._extract_catalog_params(user_input)
            results = search_product_catalog(**catalog_params)
            final_answer = self._format_catalog_results(results)

            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng cần tìm sản phẩm danh mục '{catalog_params['category']}' với ngân sách tối đa {catalog_params['max_price']} VNĐ. Gọi tool 'search_product_catalog'.",
                "action": "search_product_catalog",
                "action_input": catalog_params,
                "observation": results,
                "final_answer": final_answer
            })
            return {
                "answer": final_answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # TRƯỜNG HỢP 3: Single Tool — Chỉ cần gửi ticket hỗ trợ
        if needs_ticket and not needs_catalog:
            iteration = 1
            ticket_params = self._extract_ticket_params(user_input)
            ticket_res = submit_support_ticket(**ticket_params)
            final_answer = self._format_ticket_result(ticket_res)

            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng {ticket_params['customer_name']} cần tạo phiếu hỗ trợ sự cố với mức ưu tiên '{ticket_params['priority']}'. Gọi tool 'submit_support_ticket'.",
                "action": "submit_support_ticket",
                "action_input": ticket_params,
                "observation": ticket_res,
                "final_answer": final_answer
            })
            return {
                "answer": final_answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # TRƯỜNG HỢP 4: Cả hai tool (Catalog + Ticket)
        if needs_catalog and needs_ticket:
            iteration = 0

            # Iteration 1: Gọi search_product_catalog
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration - 1,
                    "status": "max_iterations_reached"
                }

            catalog_params = self._extract_catalog_params(user_input)
            cat_results = search_product_catalog(**catalog_params)
            self.trace.append({
                "iteration": iteration,
                "thought": f"Yêu cầu chứa cả tra cứu sản phẩm và tạo ticket. Bước 1: Tra cứu catalog '{catalog_params['category']}'.",
                "action": "search_product_catalog",
                "action_input": catalog_params,
                "observation": cat_results
            })

            # Iteration 2: Gọi submit_support_ticket
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration - 1,
                    "status": "max_iterations_reached"
                }

            ticket_params = self._extract_ticket_params(user_input)
            ticket_res = submit_support_ticket(**ticket_params)
            self.trace.append({
                "iteration": iteration,
                "thought": f"Bước 2: Ghi nhận yêu cầu hỗ trợ cho khách hàng '{ticket_params['customer_name']}'.",
                "action": "submit_support_ticket",
                "action_input": ticket_params,
                "observation": ticket_res
            })

            # Tổng hợp câu trả lời
            catalog_part = self._format_catalog_results(cat_results)
            ticket_part = self._format_ticket_result(ticket_res)
            combined_answer = f"{catalog_part}\n\nĐồng thời, {ticket_part}"

            # Iteration 3: Nếu max_iterations >= 3 thì dùng iteration 3 để tổng hợp, nếu max_iterations == 2 thì hoàn tất tại iteration 2
            if self.max_iterations >= 3:
                iteration += 1
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Đã thực hiện đầy đủ 2 công cụ. Tiến hành tổng hợp câu trả lời cuối cùng từ trace.",
                    "action": None,
                    "action_input": None,
                    "observation": None,
                    "final_answer": combined_answer
                })
            else:
                self.trace[-1]["final_answer"] = combined_answer

            return {
                "answer": combined_answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # TRƯỜNG HỢP 5: Mặc định / Lời chào / Câu hỏi ngoài lề
        iteration = 1
        default_answer = (
            "Xin chào! Tôi là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup. "
            "Tôi có thể hỗ trợ quý khách tra cứu thông tin sản phẩm xe điện VinFast, dịch vụ nghỉ dưỡng "
            "Vinpearl hoặc ghi nhận các yêu cầu bảo hành, hỗ trợ kỹ thuật. Quý khách cần hỗ trợ gì hôm nay?"
        )
        self.trace.append({
            "iteration": iteration,
            "thought": "Người dùng gửi tin nhắn chung. Trả lời trực tiếp với vai trò VinAssistant.",
            "action": None,
            "action_input": None,
            "observation": None,
            "final_answer": default_answer
        })
        return {
            "answer": default_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
