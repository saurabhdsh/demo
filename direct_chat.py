import openai
import os
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    try:
        # Connect to Key Vault
        KEY_VAULT_NAME = "kv-uais-nonprod"
        KV_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
        
        # Use environment variables for initial connection
        credential = ClientSecretCredential(
            tenant_id=os.getenv('APP_TENANT_ID'),
            client_id=os.getenv('APP_CLIENT_ID'),
            client_secret=os.getenv('APP_CLIENT_SECRET'),
            additionally_allowed_tenants=["*"]
        )
        
        # Create Key Vault client
        secret_client = SecretClient(
            vault_url=KV_URI,
            credential=credential
        )
        
        # Get Azure credentials from vault
        az_cred = ClientSecretCredential(
            tenant_id=secret_client.get_secret("secret-uais-tenant-id").value,
            client_id=secret_client.get_secret("secret-client-id-uais").value,
            client_secret=secret_client.get_secret("secret-client-secret-uais").value
        )
        
        # Get the bearer token provider
        token_provider = get_bearer_token_provider(az_cred, "https://cognitiveservices.azure.com/.default")
        
        # Azure OpenAI settings
        AZURE_OPENAI_ENDPOINT = "https://prod-1.services.unitedaistudio.uhg.com/aoai-shared-openai-prod-1"
        DEPLOYMENT_NAME = "gpt-4o_2024-05-13"  # Updated deployment name
        
        print("\nConnecting to Azure OpenAI...")
        
        # Initialize the OpenAI client
        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-06-01",  # Updated API version
            azure_deployment=DEPLOYMENT_NAME,
            azure_ad_token_provider=token_provider,
            default_headers={
                "projectId": secret_client.get_secret("secret-client-project-uais").value
            }
        )
        
        # Test chat completion
        messages = [{"role": "user", "content": "hi"}]
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        print("\nChat Completion Response:")
        print(response.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if hasattr(e, 'response'):
            print("\nFull error response:")
            print(e.response.text) 