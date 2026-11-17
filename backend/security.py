"""Bearer-авторизация для эндпоинтов приложения.

Требует заголовок Authorization: Bearer <токен> и извлекает токен.
Сопоставление с реальным ключом организаторов/прогона здесь НЕ выполняется —
это зона пользователя (ключи/токены ИИ и хакатона тестирует только он).
Здесь только обеспечиваем соответствие контракту /run: заголовок обязателен.

Использование в роуте:
    @router.post("/run")
    def run(body: RunBody, token: str = Depends(require_run_token)):
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False, description="Bearer-токен прогона")


def require_run_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """Обязательный bearer-токен. Возвращает строку токена или ошибку 401."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer <token>",
        )
    return credentials.credentials