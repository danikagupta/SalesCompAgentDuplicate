# NEXT STEPS WITH AMIT (Updated: 10/5):

*Multi-tenant capability:

Use case 1: Anonymous usage - User goes to cl3vr.ai and uses it without logging in

Use case 2: Credentialed usage - User goes to cl3vr.ai and logs in with their credentials. 

Use case 3: Dedicated Enterprise - An Enterprise company becomes a cl3vr.ai customer. They get their own tenant and makes it available to their employees. Their tenant will be company.cl3vr.ai. 

Use case 4: Partner/reseller - A Consulting company sells cl3vr.ai to their Enterprise customers and creates customer accounts under their tenant. The Enterprise customer in this case will use consulting company's tenant i.e., consultingcompany.cl3vr.ai

Functional Capabilities:

1. LangSmith traces will capture org_id, account_id, and user_name in the traces

2. Security Features for Multi-tenant Enterprise solution. See ChatGPT guidance.

3. RLHF: Add Thumb-up/Thumb-down in the Streamlit Interface to give users ability to provide feedback. The feedback gets added to LangSmith in the form of AgentState state dictionary.

4. Clean up the document database in Supabase. All sub-agents to use Supabase docs instead of Google Firestore.

5. Ability to load docs by tenant

6. Make GitHub repo private

7. Update messaging on landing page
