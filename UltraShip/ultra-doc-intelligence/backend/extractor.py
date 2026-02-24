# backend/extractor.py
import json
import re
from typing import Dict, Any, Optional
import openai
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class StructuredExtractor:
    """
    Handles structured data extraction from logistics documents
    Supports both LLM-based extraction (Groq/OpenAI) and regex fallback
    """
    
    def __init__(self, document_processor):
        """
        Initialize the extractor with document processor and API configuration
        
        Args:
            document_processor: Instance of DocumentProcessor for accessing documents
        """
        self.doc_processor = document_processor
        
        # Configure for Groq (OpenAI-compatible endpoint)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.use_api = bool(self.api_key)
        
        if self.use_api:
            # Use OpenAI SDK with Groq's endpoint
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            print(f"✅ Using Groq API with model: {self.model}")
        else:
            print("⚠️ No API key found. Using regex-based extraction only.")
    
    def extract_fields(self, file_id: str) -> Dict[str, Any]:
        """
        Extract structured shipment data from document
        
        Args:
            file_id: Unique identifier of the uploaded document
            
        Returns:
            Dictionary with extracted fields (null if not found)
        """
        # Load document
        doc = self.doc_processor.load_document(file_id)
        if not doc:
            print(f"❌ Document not found: {file_id}")
            return self.get_empty_result()
        
        # Combine all chunks for full context
        full_text = "\n".join(doc['chunks'])
        print(f"📄 Extracting from document with {len(doc['chunks'])} chunks, total length: {len(full_text)} chars")
        
        # Try API-based extraction first (if available)
        if self.use_api:
            try:
                result = self.extract_with_llm(full_text)
                if result and any(value is not None for value in result.values()):
                    print("✅ Successfully extracted data using LLM")
                    return result
                else:
                    print("⚠️ LLM extraction returned no data, falling back to regex")
            except Exception as e:
                print(f"❌ LLM extraction failed: {e}, falling back to regex")
        
        # Fallback to regex-based extraction
        return self.extract_with_regex(full_text)
    
    def extract_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use LLM (via Groq) for structured extraction
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with extracted fields
        """
        # Truncate text if too long (Groq has large context, but we'll be safe)
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
            print(f"⚠️ Text truncated to {max_chars} characters")
        
        # Enhanced prompt with examples and clear instructions
        prompt = f"""You are a logistics document extraction expert. Extract the following shipment information from the document below.

REQUIRED FIELDS (use null if not found):
1. shipment_id: Any ID number (BOL#, PRO#, Reference#, Booking#)
2. shipper: Company name of sender/shipper
3. consignee: Company name of receiver
4. pickup_datetime: Date and time of pickup (format: YYYY-MM-DD HH:MM)
5. delivery_datetime: Date and time of delivery (format: YYYY-MM-DD HH:MM)
6. equipment_type: Type of trailer/container (e.g., "53ft van", "40ft container", "Reefer")
7. mode: Transport mode ("Truck", "Rail", "Ocean", "Air", "LTL", "FTL")
8. rate: Monetary amount as number (just the number, no currency symbol)
9. currency: Currency code (USD, CAD, EUR, etc.)
10. weight: Weight as number (just the number)
11. carrier_name: Name of carrier company

EXAMPLES:
- "BOL#: 123456" → shipment_id: "123456"
- "Shipper: ABC Corp" → shipper: "ABC Corp"
- "Pickup: 2024-01-15 14:30" → pickup_datetime: "2024-01-15 14:30"
- "Rate: $2,500.00" → rate: 2500.00, currency: "USD"
- "Weight: 45000 lbs" → weight: 45000

DOCUMENT TEXT:
{text}

Return ONLY a valid JSON object with these fields. No explanations, no markdown, just pure JSON.
"""
        
        try:
            # Make API call to Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Always return valid JSON with the exact field names specified. Use null for missing fields."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
                top_p=1
            )
            
            result = response.choices[0].message.content.strip()
            print(f"📝 LLM Response: {result[:200]}...")  # Log first 200 chars
            
            # Extract JSON from response (handle potential markdown or extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean common issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                data = json.loads(json_str)
                cleaned_data = self.clean_extracted_data(data)
                
                # Validate we got at least some data
                if any(v is not None for v in cleaned_data.values()):
                    return cleaned_data
                else:
                    print("⚠️ LLM returned all null values")
                    return {}
            else:
                print("❌ No JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {result}")
            return {}
        except Exception as e:
            print(f"❌ API error: {e}")
            return {}
    
    def extract_with_regex(self, text: str) -> Dict[str, Any]:
        """
        Fallback: Use regex patterns for extraction
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with extracted fields
        """
        print("🔍 Using regex-based extraction...")
        
        data = self.get_empty_result()
        
        # Comprehensive regex patterns for logistics documents
        patterns = {
            'shipment_id': [
                r'(?:BOL|PRO|REF(?:ERENCE)?|SHIP(?:MENT)?|LOAD|BOOKING)\s*[#:;]?\s*([A-Z0-9][-A-Z0-9]{3,20})',
                r'(?:BILL OF LADING|B/L)\s*[#:;]?\s*([A-Z0-9][-A-Z0-9]{3,20})',
                r'(\d{6,20})'  # Any 6-20 digit number as last resort
            ],
            'shipper': [
                r'(?:SHIPPER|FROM|SENDER)\s*[:\n]\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$|Tel|Fax|Phone)',
                r'Shipper\s*[:\-]?\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$)',
                r'From:\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$)'
            ],
            'consignee': [
                r'(?:CONSIGNEE|TO|RECEIVER|DELIVER TO)\s*[:\n]\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$|Tel|Fax|Phone)',
                r'Consignee\s*[:\-]?\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$)',
                r'To:\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$)'
            ],
            'pickup_datetime': [
                r'(?:PICKUP|PU|PICK UP)\s*(?:DATE|TIME|DT)?\s*[:\n]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)',
                r'(?:PICKUP|PU|PICK UP)\s*(?:DATE|TIME|DT)?\s*[:\n]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)'
            ],
            'delivery_datetime': [
                r'(?:DELIVER(?:Y)?|DEL|DROP(?: OFF)?)\s*(?:DATE|TIME|DT)?\s*[:\n]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)',
                r'(?:DELIVER(?:Y)?|DEL|DROP(?: OFF)?)\s*(?:DATE|TIME|DT)?\s*[:\n]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            'equipment_type': [
                r'(?:EQUIP(?:MENT)?|TRAILER|CONTAINER)\s*[:\n]\s*([A-Za-z0-9\s\-/]+?)(?:\n|$)',
                r'(?:53[\'\"]?\s*(?:FT|FOOT)?\s*(?:VAN|REEFER|DRY)|40[\'\"]?\s*(?:FT|FOOT)?\s*(?:CONTAINER|HC)|REEFER|VAN|FLATBED)'
            ],
            'mode': [
                r'(?:MODE|SERVICE)\s*[:\n]\s*([A-Za-z]+)',
                r'\b(TL|FTL|LTL|TRUCK|RAIL|OCEAN|AIR|GROUND|EXPEDITE)\b'
            ],
            'rate': [
                r'(?:RATE|CHARGE|AMOUNT|TOTAL)\s*[:\$]?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+\.\d{2})\s*(?:USD|CAD)'
            ],
            'currency': [
                r'(?:CURRENCY)\s*[:\n]\s*([A-Z]{3})',
                r'\b(USD|CAD|EUR|GBP|MXN)\b'
            ],
            'weight': [
                r'(?:WEIGHT|WT|GROSS|NET)\s*[:\n]\s*(\d{1,5}(?:,\d{3})*(?:\.\d{1})?)\s*(?:LBS|LB|POUNDS|KG|KGS)',
                r'(\d{1,5}(?:,\d{3})*(?:\.\d{1})?)\s*(?:LBS|LB|POUNDS|KG|KGS)'
            ],
            'carrier_name': [
                r'(?:CARRIER|CARRIER NAME|TRUCKING CO|TRANSPORT)\s*[:\n]\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$|Tel|Fax|Phone)',
                r'Carrier:\s*([A-Za-z0-9\s&\.,]+?)(?:\n|$)'
            ]
        }
        
        # Apply patterns
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip() if match.groups() else match.group(0).strip()
                    
                    # Clean up the value
                    value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                    value = value.strip(':,;-\n\r\t ')  # Remove common separators
                    
                    # Convert numeric fields
                    if field in ['rate', 'weight']:
                        try:
                            # Remove commas and convert to float
                            clean_value = re.sub(r'[^\d.]', '', value)
                            if clean_value:
                                value = float(clean_value)
                        except (ValueError, TypeError):
                            value = None
                    
                    data[field] = value
                    print(f"  ✓ Extracted {field}: {value}")
                    break  # Stop after first successful pattern
        
        # Post-process to ensure data quality
        data = self.clean_extracted_data(data)
        
        # Count extracted fields
        extracted_count = sum(1 for v in data.values() if v is not None)
        print(f"📊 Regex extracted {extracted_count}/11 fields")
        
        return data
    
    def clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate extracted data
        
        Args:
            data: Raw extracted data
            
        Returns:
            Cleaned data with proper types and formats
        """
        # Define required fields with default None
        required_fields = [
            'shipment_id', 'shipper', 'consignee', 'pickup_datetime',
            'delivery_datetime', 'equipment_type', 'mode', 'rate',
            'currency', 'weight', 'carrier_name'
        ]
        
        # Ensure all fields exist
        cleaned = {}
        for field in required_fields:
            cleaned[field] = data.get(field, None)
        
        # Clean specific fields
        for field in ['shipper', 'consignee', 'carrier_name']:
            if cleaned[field] and isinstance(cleaned[field], str):
                # Remove common noise
                cleaned[field] = re.sub(r'\s+', ' ', cleaned[field]).strip()
                # Remove trailing commas, colons, etc.
                cleaned[field] = re.sub(r'[,:;]$', '', cleaned[field])
        
        # Validate and clean rate
        if cleaned['rate']:
            try:
                # Ensure it's a float
                if isinstance(cleaned['rate'], str):
                    # Remove currency symbols and commas
                    cleaned_str = re.sub(r'[^\d.]', '', cleaned['rate'])
                    cleaned['rate'] = float(cleaned_str) if cleaned_str else None
                elif isinstance(cleaned['rate'], (int, float)):
                    cleaned['rate'] = float(cleaned['rate'])
            except (ValueError, TypeError):
                cleaned['rate'] = None
        
        # Validate and clean weight
        if cleaned['weight']:
            try:
                if isinstance(cleaned['weight'], str):
                    cleaned_str = re.sub(r'[^\d.]', '', cleaned['weight'])
                    cleaned['weight'] = float(cleaned_str) if cleaned_str else None
                elif isinstance(cleaned['weight'], (int, float)):
                    cleaned['weight'] = float(cleaned['weight'])
            except (ValueError, TypeError):
                cleaned['weight'] = None
        
        # Clean currency
        if cleaned['currency'] and isinstance(cleaned['currency'], str):
            # Extract 3-letter currency code
            currency_match = re.search(r'([A-Z]{3})', cleaned['currency'].upper())
            if currency_match:
                cleaned['currency'] = currency_match.group(1)
            else:
                cleaned['currency'] = None
        
        # Format dates if possible
        for field in ['pickup_datetime', 'delivery_datetime']:
            if cleaned[field] and isinstance(cleaned[field], str):
                try:
                    # Try to parse and standardize date format
                    # This is a simplified approach - you might want more sophisticated date parsing
                    cleaned[field] = cleaned[field].strip()
                except:
                    pass  # Keep as is if parsing fails
        
        return cleaned
    
    def get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'shipment_id': None,
            'shipper': None,
            'consignee': None,
            'pickup_datetime': None,
            'delivery_datetime': None,
            'equipment_type': None,
            'mode': None,
            'rate': None,
            'currency': None,
            'weight': None,
            'carrier_name': None
        }
    
    def format_as_json(self, data: Dict[str, Any], pretty: bool = True) -> str:
        """
        Format extracted data as JSON string
        
        Args:
            data: Extracted data dictionary
            pretty: Whether to pretty-print the JSON
            
        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)
    
    def validate_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data and add confidence scores per field
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'fields_extracted': sum(1 for v in data.values() if v is not None),
            'total_fields': len(data),
            'extraction_rate': sum(1 for v in data.values() if v is not None) / len(data),
            'field_validation': {}
        }
        
        # Validate each field
        for field, value in data.items():
            field_valid = {
                'value': value,
                'valid': True,
                'issues': []
            }
            
            if value is not None:
                # Type-specific validation
                if field in ['rate', 'weight']:
                    if not isinstance(value, (int, float)):
                        field_valid['valid'] = False
                        field_valid['issues'].append(f"Expected number, got {type(value)}")
                    elif value <= 0:
                        field_valid['valid'] = False
                        field_valid['issues'].append("Value should be positive")
                
                elif field in ['pickup_datetime', 'delivery_datetime']:
                    # Basic date validation
                    if not isinstance(value, str):
                        field_valid['valid'] = False
                        field_valid['issues'].append(f"Expected string, got {type(value)}")
                
                elif field == 'currency':
                    if not isinstance(value, str) or len(value) != 3:
                        field_valid['valid'] = False
                        field_valid['issues'].append("Currency should be 3-letter code")
            
            validation['field_validation'][field] = field_valid
        
        return validation


# Optional: Add a main function for testing
if __name__ == "__main__":
    # Simple test code
    print("🧪 Testing StructuredExtractor...")
    
    # Mock document processor for testing
    class MockDocProcessor:
        def load_document(self, file_id):
            return {
                'chunks': [
                    "BOL#: 123456789",
                    "Shipper: ABC Manufacturing Inc.",
                    "Consignee: XYZ Distribution LLC",
                    "Pickup: 2024-01-15 08:00",
                    "Delivery: 2024-01-17 14:30",
                    "Equipment: 53ft Dry Van",
                    "Mode: FTL",
                    "Rate: $2,450.00 USD",
                    "Weight: 42000 lbs",
                    "Carrier: Speedy Transport Co."
                ]
            }
    
    # Test extractor
    extractor = StructuredExtractor(MockDocProcessor())
    result = extractor.extract_fields("test123")
    
    print("\n📊 Extraction Result:")
    print(json.dumps(result, indent=2))
    
    # Validate
    validation = extractor.validate_extraction(result)
    print(f"\n✅ Validation: {validation['fields_extracted']}/{validation['total_fields']} fields extracted")