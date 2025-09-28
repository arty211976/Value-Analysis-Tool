#!/usr/bin/env python3
"""
LLM Configuration for Different Roles
Easy way to configure different LLMs for different tasks
"""

# LLM Configuration Options
LLM_SETUPS = {
    "separated": {
        "description": "Use different LLMs for different roles (recommended)",
        "tool_handling": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "provider": "openai"
        },
        "reasoning": {
            "model": "deepseek-r1:7b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "temperature": 0,
            "provider": "ollama"
        }
    },
    "unified_openai": {
        "description": "Use OpenAI GPT-4o-mini for all roles (current default)",
        "tool_handling": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "provider": "openai"
        },
        "reasoning": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "provider": "openai"
        }
    },
    "unified_openai_premium": {
        "description": "Use OpenAI GPT-4o for all roles (premium)",
        "tool_handling": {
            "model": "gpt-4o",
            "temperature": 0,
            "provider": "openai"
        },
        "reasoning": {
            "model": "gpt-4o",
            "temperature": 0,
            "provider": "openai"
        }
    },
    "unified_ollama": {
        "description": "Use Ollama Deepseek 7B for all roles",
        "tool_handling": {
            "model": "deepseek-r1:7b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "temperature": 0,
            "provider": "ollama"
        },
        "reasoning": {
            "model": "deepseek-r1:7b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "temperature": 0,
            "provider": "ollama"
        }
    }
}

# Current active configuration
ACTIVE_CONFIG = "separated"

def get_llm_config(config_name=None):
    """
    Get LLM configuration for the specified setup
    """
    if config_name is None:
        config_name = ACTIVE_CONFIG
    
    if config_name not in LLM_SETUPS:
        print(f"Configuration '{config_name}' not found. Using 'separated'.")
        config_name = "separated"
    
    return LLM_SETUPS[config_name]

def create_llm_from_config(config, llm_type):
    """
    Create LLM instance from configuration
    """
    from crewai import LLM
    
    llm_config = config[llm_type]
    
    if llm_config["provider"] == "openai":
        return LLM(
            model=llm_config["model"],
            temperature=llm_config["temperature"]
        )
    elif llm_config["provider"] == "ollama":
        try:
            # For LiteLLM with Ollama, we need to use the ollama/ prefix
            ollama_model_name = f"ollama/{llm_config['model']}"
            return LLM(
                model=ollama_model_name,
                temperature=llm_config["temperature"]
            )
        except Exception as e:
            print(f"Warning: Failed to create Ollama LLM: {e}")
            print("Falling back to GPT-4o-mini")
            # Fallback to OpenAI
            return LLM(
                model="gpt-4o-mini",
                temperature=llm_config["temperature"]
            )
    else:
        raise ValueError(f"Unknown provider: {llm_config['provider']}")

def get_llm_instances(config_name=None):
    """
    Get both tool_handling and reasoning LLM instances
    """
    config = get_llm_config(config_name)
    
    tool_handling_llm = create_llm_from_config(config, "tool_handling")
    reasoning_llm = create_llm_from_config(config, "reasoning")
    
    return tool_handling_llm, reasoning_llm

def print_available_configs():
    """
    Print all available LLM configurations
    """
    print("🔧 Available LLM Configurations")
    print("=" * 50)
    
    for name, config in LLM_SETUPS.items():
        status = " ACTIVE" if name == ACTIVE_CONFIG else ""
        print(f"\n{name.upper()}{status}")
        print(f"Description: {config['description']}")
        print(f"Tool Handling: {config['tool_handling']['model']} ({config['tool_handling']['provider']})")
        print(f"Reasoning: {config['reasoning']['model']} ({config['reasoning']['provider']})")

def print_current_config():
    """Print the current LLM configuration"""
    config = get_llm_config()
    print("Current LLM Configuration")
    print("=" * 40)
    print(f"Active Config: {ACTIVE_CONFIG}")
    print(f"Tool Handling LLM: {config['tool_handling']['model']}")
    print(f"Reasoning LLM: {config['reasoning']['model']}")
    print(f"Tool Handling Base URL: {config['tool_handling'].get('base_url', 'N/A')}")
    print(f"Reasoning Base URL: {config['reasoning'].get('base_url', 'N/A')}")
    print("=" * 40)

def test_ollama_connection():
    """
    Test connection to Ollama API
    """
    import requests
    
    config = get_llm_config()
    if config['reasoning']['provider'] == 'ollama':
        url = config['reasoning']['base_url'].replace('/v1', '/api/generate')
        data = {
            "model": config['reasoning']['model'],
            "prompt": "Say hello from deepseek!",
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            print(f"Ollama Connection Test:")
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Ollama is running and accessible")
                result = response.json()
                print(f"  Response: {result.get('response', 'No response')[:100]}...")
            else:
                print(f"  Ollama connection failed: {response.text}")
        except Exception as e:
            print(f"  Ollama connection error: {e}")
    else:
        print("Ollama not configured in current setup")

if __name__ == "__main__":
    print_current_config()
    print("\n" + "=" * 50)
    test_ollama_connection()
    print("\n" + "=" * 50)
    print_available_configs() 