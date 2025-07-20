"""
Interactive demo script for the multi-agent system.
This script provides a command-line interface to interact with the agents.
"""

from multi_agent_system import MultiAgentSystem
import json


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("-" * 60)


def display_agent_response(response):
    """Display agent response in a formatted way"""
    print(f"\n🤖 Agent: {response.agent_name}")
    print_separator()
    
    try:
        # Try to parse and display JSON content nicely
        content_data = json.loads(response.content)
        for key, value in content_data.items():
            if isinstance(value, list):
                print(f"{key.upper()}:")
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"{key.upper()}: {value}")
    except (json.JSONDecodeError, TypeError):
        print(response.content)
    
    print(f"\n📊 Metadata: {response.metadata}")


def main():
    print("🚀 Interactive Multi-Agent System Demo")
    print("=" * 50)
    
    # Initialize system
    system = MultiAgentSystem()
    
    while True:
        print("\n🎯 Choose an option:")
        print("1. Run complete multi-agent workflow")
        print("2. Run individual agent")
        print("3. List available agents")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            topic = input("\n📝 Enter a topic for the multi-agent workflow: ").strip()
            if topic:
                print(f"\n🚀 Running complete workflow for: '{topic}'")
                try:
                    results = system.run_workflow(topic)
                    
                    print("\n" + "="*60)
                    print("📋 DETAILED RESULTS")
                    print("="*60)
                    
                    for phase, response in results.items():
                        if hasattr(response, 'agent_name'):
                            display_agent_response(response)
                            
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Please enter a valid topic.")
        
        elif choice == "2":
            agents = system.list_agents()
            print(f"\n🤖 Available agents: {', '.join(agents)}")
            
            agent_name = input("Enter agent name: ").strip().lower()
            if agent_name in agents:
                input_text = input("Enter input text: ").strip()
                context = input("Enter context (optional): ").strip()
                
                try:
                    response = system.run_single_agent(agent_name, input_text, context)
                    display_agent_response(response)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print(f"❌ Invalid agent name. Choose from: {agents}")
        
        elif choice == "3":
            agents = system.list_agents()
            print(f"\n🤖 Available agents:")
            for i, agent in enumerate(agents, 1):
                print(f"  {i}. {agent}")
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
