// One-click Playground examples. Each fills the state (text, or JSON when an object), the instructions and the
// questions; the shapes follow the /v1/systemone contract (docs/SYSTEM_ONE.md). `multi` and `act` are basal-1.5
// extensions of that contract.

export const TEMPLATES = [
  {
    id: 'moderation',
    name: 'Content moderation',
    description: 'True/false checks on a user comment.',
    state: 'Great tutorial! If anyone wants the full course for free, DM me on telegram @cheapcourses, I sell all paid courses for $5.',
    instructions: 'You moderate comments on a programming tutorial site.',
    questions: {
      spam: {
        type: 'noul',
        instructions: 'Is the comment spam or unsolicited advertising?',
        criteria: { true: 'Promotes an unrelated product, service or contact channel', false: 'A genuine reaction or question' },
      },
      abusive: {
        type: 'noul',
        instructions: 'Does the comment insult, harass or threaten anyone?',
      },
    },
  },
  {
    id: 'routing',
    name: 'Support ticket routing',
    description: 'Pick a department from a list and flag urgency.',
    state: 'After the latest update the app crashes on start and our whole team cannot work. We need this fixed today.',
    instructions: '',
    questions: {
      department: {
        type: 'choice',
        instructions: 'Which department should handle this ticket? If information is missing, choose other.',
        criteria: {
          returns: 'Refund, complaint or product replacement',
          support: 'Help with configuring or using the product',
          sales: 'Questions before a purchase',
          other: 'Anything else, or not enough information',
        },
      },
      urgent: {
        type: 'noul',
        instructions: 'Does the ticket need an urgent reaction?',
        criteria: { true: 'Describes a security threat or an outage that blocks work', false: 'A regular request' },
      },
    },
  },
  {
    id: 'sentiment',
    name: 'Sentiment with rubric',
    description: 'Ordered 5-level scale with a description of every level.',
    state: 'The headphones sound good and the battery lasts all week, but the ear cushions started peeling after a month.',
    instructions: '',
    questions: {
      sentiment: {
        type: 'score',
        instructions: 'How satisfied is the customer overall?',
        criteria: [
          'Very negative: angry, wants a refund or warns others',
          'Negative: disappointed, more drawbacks than advantages',
          'Mixed: clear advantages and clear drawbacks',
          'Positive: satisfied, minor complaints at most',
          'Very positive: enthusiastic, would recommend',
        ],
      },
    },
  },
  {
    id: 'lead',
    name: 'Lead qualification',
    description: 'Structured CRM record: fit score, budget check and next step.',
    state: {
      company: 'Northwind Logistics',
      employees: 420,
      industry: 'Freight forwarding',
      contact_role: 'Head of Operations',
      message: 'We process about 3,000 customs documents a month by hand and want to automate it this quarter. Budget is approved.',
      source: 'Webinar sign-up',
    },
    instructions: 'We sell document automation software to mid-sized companies (100–2,000 employees).',
    questions: {
      fit: {
        type: 'score',
        instructions: 'How well does the lead match our ideal customer profile?',
        criteria: ['Not a fit', 'Weak fit', 'Good fit', 'Ideal fit'],
      },
      budget_confirmed: {
        type: 'noul',
        instructions: 'Does the lead state that budget is available?',
      },
      next_step: {
        type: 'choice',
        instructions: 'What should sales do next?',
        criteria: {
          demo: 'Book a product demo within a week',
          nurture: 'Add to the newsletter and follow up next quarter',
          disqualify: 'Close the lead',
        },
      },
    },
  },
  {
    id: 'invoice',
    name: 'Invoice fraud check',
    description: 'Risk flag plus a cost-based action (approve, hold or human review).',
    state: {
      vendor: 'ACME Office Supplies Ltd',
      vendor_known_since: '2019-04',
      invoice_number: 'INV-2026-1187',
      amount: 18450.0,
      currency: 'EUR',
      usual_monthly_amount: 1200.0,
      bank_account_changed: true,
      note: 'Please pay to our new account urgently, the old one is closed.',
    },
    instructions: 'You review supplier invoices before payment.',
    questions: {
      suspicious: {
        type: 'noul',
        instructions: 'Does the invoice show signs of payment fraud?',
        criteria: {
          true: 'Unusual amount, changed bank details or pressure to pay quickly',
          false: 'Consistent with the usual relationship with this vendor',
        },
      },
      payment: {
        type: 'act',
        base: 'noul',
        instructions: 'Is the invoice legitimate and safe to pay?',
        criteria: { true: 'Legitimate', false: 'Not legitimate' },
        costs: {
          pay: { true: 0, false: 50 },
          hold: { true: 5, false: 0 },
          review: { true: 2, false: 2 },
        },
        defer_action: 'review',
      },
    },
  },
  {
    id: 'all-types',
    name: 'All question types',
    description: 'One question of every type, including multi-select and action.',
    state: 'The phone arrived with a cracked screen two days late. I want my money back.',
    instructions: '',
    questions: {
      department: {
        type: 'choice',
        instructions: 'Which department should handle this ticket?',
        criteria: { returns: 'Refund or complaint', support: 'Technical help', sales: 'Pre-purchase questions', other: 'Anything else' },
      },
      urgent: {
        type: 'noul',
        instructions: 'Does the ticket describe a security threat or an outage that blocks work?',
      },
      severity: {
        type: 'score',
        instructions: 'Rate the impact on using the product.',
        criteria: ['No impact', 'Minor issues; core functions work', 'Major limits; some core functions fail', 'Product is unusable'],
      },
      tags: {
        type: 'multi',
        instructions: 'Select every topic present in the ticket.',
        criteria: { damage: 'Product damage', refund: 'Refund request', delivery: 'Delivery problem' },
        threshold: 0.7,
      },
      refund_action: {
        type: 'act',
        base: 'noul',
        instructions: 'Does the complaint meet the refund conditions stated in the ticket?',
        criteria: { true: 'Meets the refund conditions', false: 'Does not meet them' },
        costs: {
          approve: { true: 0, false: 10 },
          reject: { true: 20, false: 0 },
          human: { true: 2, false: 2 },
        },
        defer_action: 'human',
      },
    },
  },
];
