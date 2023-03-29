from __future__ import annotations

from linecache import getline
from logging import Logger

logger = Logger(__name__)

INSTALL_COMMANDS = {
    "yolov5": "`poetry install -E yolov5`",
    "mmcv": "`poe install-mmlab-all`",
    "mmdet": "`poe install-mmlab-all`",
}


class SoftImport:
    def __init__(self, silent: bool = False) -> None:
        """Context manager for soft imports.

        Args:
            silent (bool, optional): Whether to suppress warnings. Errors are always displayed
                Defaults to False.
            import_source (str, optional): The name of the module checking for soft imports.
                Defaults to __name__.
        """
        self.silent = silent
        self.import_source = None
        self.modules = []

    def __enter__(self):
        return self

    @staticmethod
    def _get_base_module(module_name: str):
        return module_name.split(".")[0]

    @property
    def base_modules(self) -> list[str]:
        return [self._get_base_module(module) for module in self.modules]

    def _get_module_name(self, code_line: str) -> str:
        code_line = code_line.strip().replace("  ", " ")
        if code_line.startswith("from "):
            return code_line.split("from ")[1].split(" import")[0]

        if code_line.startswith("import "):
            return code_line.split("import ")[1].split(" ")[0]

        raise ValueError(f"Code line is not a valid import statement: {code_line}")

    @staticmethod
    def _get_code(exc_tb) -> str:
        ex_line = exc_tb.tb_lineno
        ex_file = exc_tb.tb_frame.f_code.co_filename
        line_code = getline(ex_file, ex_line).strip()
        return line_code

    def _import_error_str(self, module_name: str) -> str:
        UNKNOWN_MODULE_STR = "Unknown module: You can try `poetry install --all-extras`"
        return f"Module `{self.import_source}` requires the module `{module_name}`. You can install it using this command: {INSTALL_COMMANDS.get(module_name, UNKNOWN_MODULE_STR)}."

    def _import_error_msg(self) -> str:
        MULTIPLE_EXTRAS_STR = 'To install multiple extras run `poetry install -E "extra1 extra2"`'
        EXTRAS_INFO_STR = "For more info about optional dependencies check-out the documentation at https://datasentics.github.io/qi-lib/installation.html."

        error_strs = []
        for module in self.base_modules:
            error_str = self._import_error_str(module)
            error_strs.append(error_str)

        error_strs.extend([MULTIPLE_EXTRAS_STR, EXTRAS_INFO_STR])
        return "\n".join(error_strs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if isinstance(exc_val, ImportError):
            line = self._get_code(exc_tb)
            module = self._get_module_name(line)
            self.modules.append(module)

            # Get the module name for the error message
            self.import_source = exc_tb.tb_frame.f_globals["__name__"]

            if not self.silent:
                logger.warning(self._import_error_str(self._get_base_module(module)))

            return True
        elif exc_type is not None:
            raise exc_val

        return True

    def check(self) -> None:
        """Check if there are any missing modules. Should be placed before any usage of
        symbols imported from optional dependencies.

        Raises:
            ModuleNotFoundError: One or more missing modules in the environment.
        """
        if len(self.modules) != 0:
            logger.error(self._import_error_msg())
            raise ModuleNotFoundError()
